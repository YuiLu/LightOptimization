using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditor.IMGUI.Controls;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace LightingMvp.Editor
{
    /// <summary>Scrubs and plays recorded optimizer states without rerunning the optimization.</summary>
    public sealed class LightingDebugWindow : EditorWindow
    {
        [SerializeField] private int activeTab,seed=1234;
        [SerializeField] private Bounds effectBounds=new Bounds(new Vector3(0,1,0),new Vector3(2,2,2));
        [SerializeField] private bool selectedLightsOnly,boundsInitialized;
        [SerializeField] private float positionSigma=0.15f,intensitySigma=0.20f,directionSigma=12,colorSigma;
        [SerializeField] private int fieldResolution=30,fieldMode,fieldExperiment,fieldDisplayMode,fieldColormap=1,fieldQuantity;
        [SerializeField] private bool showIrradiance,irradianceAdvanced,autoFieldUpdate=true,autoFieldNormalize=true,logFieldScale=true,rgbFieldMode;
        [SerializeField] private float fieldContrast=20,fieldMarkerScale=1,manualFieldMax=0.0005f,fieldOverlayOpacity=1;
        private bool fieldDirty=true;
        private double nextFieldUpdate;
        private string fieldLightSignature="",fieldGeometrySignature="";
        private SurfaceSampleSet fieldSamples;
        private DiffuseRadiosity fieldTransport;
        private Vector3[] fieldValues;
        private float fieldMax;
        private Texture2D fieldLegend;
        private readonly System.Collections.Generic.List<FieldOverlay> fieldOverlays=new System.Collections.Generic.List<FieldOverlay>();
        private GameObject fieldOverlayRoot;
        private Material fieldOverlayMaterial;
        private sealed class FieldOverlay { public Mesh mesh;public int start,count; }
        private readonly string[] fieldModes={ "Current scene lights","Target field","Absolute error |current-target|" };
        private readonly BoxBoundsHandle boundsHandle=new BoxBoundsHandle();
        private readonly System.Collections.Generic.List<LightState> perturbationBaseline=new System.Collections.Generic.List<LightState>();
        private int lastPerturbedCount=-1;

        private sealed class LightState { public Light light;public Vector3 position;public Quaternion rotation;public float intensity;public Color color; }

        private const string ScenePath=CornellBoxMvpEditor.ScenePath;
        private const int RadiosityQuantity=2;
        private static string OutputRoot=>CornellBoxMvpEditor.OutputRoot;
        private readonly string[] files={ "point_position_intensity.json","spot_position_direction.json","three_point_position_intensity.json","three_point_joint_cct.json",CornellBoxMvpEditor.RadiosityExperiment+".json" };
        private readonly string[] labels={ "Point: position + intensity","Spot: position + direction","Three points: position + intensity","Three points: position + intensity + CCT","Three points: multi-bounce GI + CCT" };
        private bool RadiosityField=>fieldQuantity==RadiosityQuantity;
        private static readonly string[] replayViews={ "All lights","Selected light only" };
        [SerializeField] private int experiment=-1,selectedReplayLight;
        [SerializeField] private bool isolateReplayLight;
        private CornellBoxMvpEditor.ExperimentReport[] reports;
        private int index;
        private Vector2 scrollPosition;
        private float speed=45;
        private bool playing,showTarget=true;
        private double nextUpdate;
        private CornellSceneRig replayRig;
        internal bool HasScenePreview => fieldOverlayRoot != null || replayRig != null;
        private Light candidate=>replayRig!=null&&replayRig.Count>0?replayRig[0]:null;

        [MenuItem("Lighting MVP/Debug/Legacy field and replay")]
        public static void Open()
        {
            var window=GetWindow<LightingDebugWindow>();window.titleContent=new GUIContent("Lighting Debug");window.Show();
        }

        public static void BatchPreviewSmoke()
        {
            try
            {
                EditorSceneManager.OpenScene(ScenePath);
                VerifyCornellPlayback();EditorApplication.Exit(0);
            }
            catch(Exception e){Debug.LogException(e);EditorApplication.Exit(2);}
        }

        internal static void VerifyCornellPlayback()
        {
            using var baseline=new CornellSceneRig(SceneManager.GetActiveScene());
            var states=new string[baseline.Count];
            for(int i=0;i<baseline.Count;i++)states[i]=LightSignature(baseline[i]);
            var window=CreateInstance<LightingDebugWindow>();
            try
            {
                if(RecordedLightCount(window.reports[window.experiment])!=3||window.isolateReplayLight)
                    throw new Exception("Replay did not default to a recorded multi-light rig in all-lights mode.");
                for(int i=0;i<window.files.Length;i++)
                {
                    window.experiment=i;
                    var history=window.reports[i].history;
                    int frame=Mathf.Min(75,history.Count-3);
                    window.index=frame;window.ApplyCurrent();
                    AssertFrame(frame,-1);
                    if(history[frame].lights.Length==1)continue;

                    window.playing=true;window.nextUpdate=0;window.Tick();
                    AssertFrame(++frame,-1);
                    window.SetReplayView(true,1);
                    AssertFrame(frame,1);
                    if(!window.playing)throw new Exception("Changing replay mode paused playback.");
                    window.nextUpdate=0;window.Tick();
                    AssertFrame(++frame,1);
                    window.SetReplayView(true,2);
                    AssertFrame(frame,2);
                    window.SetReplayView(false,2);
                    AssertFrame(frame,-1);
                    window.playing=false;
                }
                window.experiment=3;window.ApplyCurrent();
                window.showIrradiance=true;window.fieldMode=1;window.fieldExperiment=3;window.fieldQuantity=1;window.RecomputeIrradiance();
                if(window.fieldOverlays.Count!=7||window.fieldMax<=0)throw new Exception("Multi-light radiance overlay mismatch.");
                CornellBoxMvpEditor.Capture("three_point_target_radiance_heatmap.png");
                window.SetReplayView(true,2);
                AssertIsolatedOverlay();
                window.experiment=Array.IndexOf(window.files,CornellBoxMvpEditor.RadiosityExperiment+".json");
                window.ApplyCurrent();
                window.fieldQuantity=RadiosityQuantity;window.fieldMode=1;window.RecomputeIrradiance();
                var giReport=window.reports[window.experiment];
                if(!giReport.usesRadiosity||window.fieldTransport==null||window.fieldMax<=0||window.fieldOverlays.Count!=0)
                    throw new Exception("Multi-bounce replay did not use the radiosity patch overlay.");
                var selectedTarget=new[]{TargetLights(giReport)[2]};
                var giSamples=window.fieldSamples;
                var expectedGi=AnalyticLighting.EvaluateField(selectedTarget,giSamples,UnityVisibility.Evaluate(selectedTarget,giSamples),true,window.fieldTransport);
                if(AnalyticLighting.FieldNrmse(window.fieldValues,expectedGi,giSamples)>1e-6f)
                    throw new Exception("Multi-bounce overlay did not isolate the selected target light.");
                window.StopReplay();
                for(int i=0;i<states.Length;i++)if(states[i]!=LightSignature(baseline[i]))throw new Exception("Replay did not restore the original Cornell light state.");
                Debug.Log("Cornell replay smoke passed: all experiments, synchronized and isolated playback, direct and multi-bounce overlays, and scene restoration.");
            }
            finally{DestroyImmediate(window);}

            void AssertFrame(int frame,int onlyLight)
            {
                if(window.index!=frame||window.replayRig==null)throw new Exception("Replay did not stay on the expected frame.");
                LightSpec[] expected=window.reports[window.experiment].history[frame].lights;
                for(int l=0;l<window.replayRig.Count;l++)
                {
                    Light actual=window.replayRig[l];
                    bool visible=l<expected.Length&&(onlyLight<0||l==onlyLight);
                    if(actual.gameObject.activeSelf!=visible||actual.isActiveAndEnabled!=visible)
                        throw new Exception("Replay did not activate the correct light slots.");
                    if(visible&&(Vector3.Distance(actual.transform.position,expected[l].position)>1e-6f||
                        Mathf.Abs(actual.intensity-expected[l].intensity)>1e-6f||Mathf.Abs(actual.colorTemperature-expected[l].cctKelvin)>0.01f))
                        throw new Exception("Replay light state mismatch.");
                }
            }

            void AssertIsolatedOverlay()
            {
                // Deliberately keep the overlay's all-lights target on another experiment.
                window.fieldExperiment=0;
                var report=window.reports[window.experiment];
                var samples=window.fieldSamples;
                var frame=report.history[window.index].lights;
                Vector3[] current=Radiance(frame[2],false),target=Radiance(TargetLights(report)[2],report.usesCct);
                for(int mode=0;mode<3;mode++)
                {
                    window.fieldMode=mode;window.RecomputeIrradiance();
                    if(window.fieldValues?.Length!=samples.Count)throw new Exception("Isolated overlay field is missing.");
                    for(int i=0;i<samples.Count;i++)
                    {
                        Vector3 d=current[i]-target[i];
                        Vector3 expected=mode==0?current[i]:mode==1?target[i]:new Vector3(Mathf.Abs(d.x),Mathf.Abs(d.y),Mathf.Abs(d.z));
                        if(Vector3.Distance(window.fieldValues[i],expected)>Mathf.Max(1e-7f,expected.magnitude*1e-4f))
                            throw new Exception("Isolated overlay is not the selected light's field (mode "+mode+").");
                    }
                }

                Vector3[] Radiance(LightSpec spec,bool cct)
                {
                    var rig=new[]{spec};
                    var mask=report.usesVisibility?UnityVisibility.Evaluate(rig,samples):null;
                    return AnalyticLighting.DiffuseRadiance(AnalyticLighting.Evaluate(rig,samples,mask,cct),samples);
                }
            }

            static string LightSignature(Light light)=>light.gameObject.activeSelf+"/"+light.transform.localPosition.ToString("R")+"/"+
                light.transform.localRotation.ToString("R")+"/"+EditorJsonUtility.ToJson(light);
        }

        public static void BatchPerturbationSmoke() { try { var window=GetWindow<LightingDebugWindow>();window.experiment=0;window.OpenExperimentScene();window.activeTab=1;window.effectBounds=new Bounds(new Vector3(0,1,0),new Vector3(2,2,2));if(window.GetLightsInScope().Count!=1)throw new Exception("Bounding-box scope mismatch");window.seed=1234;Vector3 before=window.candidate.transform.position;window.ApplyRandomPerturbation();Vector3 perturbed=window.candidate.transform.position;if(Vector3.Distance(before,perturbed)<1e-6f)throw new Exception("Perturbation did not move the light");window.RestorePerturbationBaseline();if(Vector3.Distance(before,window.candidate.transform.position)>1e-6f)throw new Exception("Perturbation restore mismatch");window.seed=1234;window.ApplyRandomPerturbation();if(Vector3.Distance(perturbed,window.candidate.transform.position)>1e-6f)throw new Exception("Perturbation is not reproducible");window.RestorePerturbationBaseline();window.effectBounds=new Bounds(new Vector3(10,10,10),Vector3.one);if(window.GetLightsInScope().Count!=0)throw new Exception("Bounding-box exclusion mismatch");window.Close();Debug.Log("Lighting perturbation smoke test passed");EditorApplication.Exit(0); } catch(Exception e){Debug.LogException(e);EditorApplication.Exit(2);} }

        public static void BatchIrradianceSmoke() { try { var window=GetWindow<LightingDebugWindow>();window.OpenExperimentScene();window.showIrradiance=true;window.fieldMode=0;window.RecomputeIrradiance();if(window.fieldSamples.Count<=5*(window.fieldResolution+1)*(window.fieldResolution+1)||window.fieldValues.Length!=window.fieldSamples.Count||window.fieldMax<=0)throw new Exception("Current irradiance visualization data mismatch");if(window.fieldDisplayMode==0&&(window.fieldOverlays.Count!=7||window.fieldOverlays[0].mesh.colors.Length==0))throw new Exception("Continuous irradiance mesh was not generated");window.fieldMode=1;window.fieldExperiment=0;window.RecomputeIrradiance();if(window.fieldMax<=0)throw new Exception("Target irradiance visualization data mismatch");window.fieldMode=2;window.RecomputeIrradiance();if(window.fieldMax<=0)throw new Exception("Error irradiance visualization data mismatch");window.ExportDisplayedField();window.Close();Debug.Log("Irradiance visualization smoke test passed");EditorApplication.Exit(0); } catch(Exception e){Debug.LogException(e);EditorApplication.Exit(2);} }

        public static void OpenContinuousIrradianceVisual() { var window=GetWindow<LightingDebugWindow>();window.OpenExperimentScene();window.showIrradiance=true;window.fieldDisplayMode=0;window.fieldMode=1;window.fieldExperiment=0;window.RecomputeIrradiance();EditorApplication.ExecuteMenuItem("Window/General/Game"); }

        internal static void EndCornellPreviews()
        {
            foreach(var window in Resources.FindObjectsOfTypeAll<LightingDebugWindow>()){window.StopReplay();window.DestroyFieldOverlays();window.fieldDirty=true;}
        }

        internal static void RefreshCornellReports()
        {
            foreach(var window in Resources.FindObjectsOfTypeAll<LightingDebugWindow>()){window.LoadReports();window.fieldDirty=true;window.Repaint();}
        }

        private void LoadReports()
        {
            reports=new CornellBoxMvpEditor.ExperimentReport[files.Length];
            for(int i=0;i<files.Length;i++)
            {
                string path=Path.Combine(OutputRoot,files[i]);
                if(File.Exists(path))reports[i]=JsonUtility.FromJson<CornellBoxMvpEditor.ExperimentReport>(File.ReadAllText(path));
            }
            if(experiment<0||experiment>=reports.Length||RecordedLightCount(reports[experiment])==0)
            {
                experiment=Array.FindIndex(reports,report=>RecordedLightCount(report)>1);
                if(experiment<0)experiment=Mathf.Max(0,Array.FindIndex(reports,report=>RecordedLightCount(report)>0));
            }
        }

        private static int RecordedLightCount(CornellBoxMvpEditor.ExperimentReport report)=>
            report?.history!=null&&report.history.Count>0?report.history[0].lights?.Length??0:0;

        private static LightSpec[] TargetLights(CornellBoxMvpEditor.ExperimentReport report)=>
            report.targetRig!=null&&report.targetRig.Length>0?report.targetRig:new[]{report.target};

        private int VisibleReplayLight(int lightCount)=>
            isolateReplayLight&&lightCount>1?Mathf.Clamp(selectedReplayLight,0,lightCount-1):-1;

        private int IsolatedPreviewLight()
        {
            if(activeTab!=0||replayRig==null)return -1;
            var history=reports?[experiment]?.history;
            return history!=null&&index>=0&&index<history.Count?VisibleReplayLight(history[index].lights?.Length??0):-1;
        }

        private static string ReplayLightLabel(int light)=>"Light "+(light+1);

        private void SetReplayView(bool isolate,int selectedLight)
        {
            isolateReplayLight=isolate;selectedReplayLight=selectedLight;
            if(replayRig!=null)ApplyCurrent();
        }

        private void OnEnable()
        {
            LoadReports();BuildFieldLegend();DestroyFieldOverlays();fieldDirty=true;
            EditorApplication.update+=Tick;
            SceneView.duringSceneGui+=DrawSceneMarkers;
        }

        private void OnDisable()
        {
            EditorApplication.update-=Tick;SceneView.duringSceneGui-=DrawSceneMarkers;
            if(fieldLegend!=null)UnityEngine.Object.DestroyImmediate(fieldLegend);DestroyFieldOverlays();
            StopReplay();
        }

        private void OnGUI()
        {
            int tab=GUILayout.Toolbar(Mathf.Clamp(activeTab,0,1),new[]{"Iteration Replay","Light Perturbation"});
            if(tab!=activeTab){StopReplay();activeTab=tab;scrollPosition=Vector2.zero;}
            using(var scroll=new EditorGUILayout.ScrollViewScope(scrollPosition))
            {
                scrollPosition=scroll.scrollPosition;
                EditorGUILayout.Space();bool enabled=EditorGUILayout.ToggleLeft("Show Irradiance",showIrradiance,EditorStyles.boldLabel);
                if(enabled!=showIrradiance){showIrradiance=enabled;fieldDirty=true;if(!showIrradiance){DestroyFieldOverlays();fieldValues=null;}RefreshEditorViews();}
                bool cornell=SceneManager.GetActiveScene().path==ScenePath;
                if(showIrradiance&&!cornell)EditorGUILayout.HelpBox("Use Native Irradiance Preview for this scene. The analytic overlays and recorded targets below are Cornell-specific.",MessageType.Info);
                else if(showIrradiance)DrawIrradianceControls();
                if(GUILayout.Button("Open Native Irradiance Preview"))NativeIrradianceWindow.Open();
                EditorGUILayout.Space();if(activeTab==0)DrawIterationTab();else DrawPerturbationTab();
            }
        }

        private void DrawIterationTab()
        {
            if(SceneManager.GetActiveScene().path!=ScenePath)
            {
                EditorGUILayout.HelpBox("迭代预览使用已保存的 Cornell Box 实验场景。",MessageType.Info);
                if(GUILayout.Button("Open Cornell Box Experiment"))OpenExperimentScene();
                return;
            }
            if(GUILayout.Button("Reload recorded runs")){StopReplay();LoadReports();fieldDirty=true;}
            EditorGUI.BeginChangeCheck();
            experiment=EditorGUILayout.Popup("Experiment",experiment,labels);
            if(EditorGUI.EndChangeCheck()){index=0;playing=false;ApplyCurrent();}
            var report=reports[experiment];
            if(RecordedLightCount(report)==0)
            {
                EditorGUILayout.HelpBox("No legacy Cornell history found. For new fits, use Lighting MVP > Optimize Lighting (Unity RGB).",MessageType.Warning);return;
            }
            int count=report.history.Count;index=Mathf.Clamp(index,0,count-1);
            DrawReplayView(report.history[index].lights.Length);
            if(replayRig==null&&GUILayout.Button("Preview recorded state"))ApplyCurrent();
            if(replayRig!=null&&GUILayout.Button("Restore scene lights"))StopReplay();
            EditorGUI.BeginChangeCheck();
            index=EditorGUILayout.IntSlider("Recorded state",index,0,count-1);
            if(EditorGUI.EndChangeCheck()){playing=false;ApplyCurrent();}
            EditorGUILayout.BeginHorizontal();
            if(GUILayout.Button("|<")){index=0;playing=false;ApplyCurrent();}
            if(GUILayout.Button("<")){index=Mathf.Max(0,index-1);playing=false;ApplyCurrent();}
            bool wasPlaying=playing;
            playing=GUILayout.Toggle(playing,playing?"Pause":"Play","Button",GUILayout.Width(80));
            if(playing&&!wasPlaying){if(index==count-1)index=0;ApplyCurrent();}
            if(GUILayout.Button(">")){index=Mathf.Min(count-1,index+1);playing=false;ApplyCurrent();}
            if(GUILayout.Button(">|")){index=count-1;playing=false;ApplyCurrent();}
            EditorGUILayout.EndHorizontal();
            speed=EditorGUILayout.Slider("States / second",speed,1,120);
            showTarget=EditorGUILayout.Toggle("Show target markers",showTarget);
            EditorGUILayout.Space();
            OptimizationHistoryEntry entry=report.history[index];
            string phase=entry.step<=report.steps?"Adam":"L-BFGS accepted step";
            EditorGUILayout.LabelField(phase,"Step "+entry.step);
            EditorGUILayout.LabelField("Rig loss",entry.loss.ToString("E6"));
            EditorGUILayout.LabelField("Rig field NRMSE",entry.nrmse.ToString("P6"));
            if(report.usesRadiosity)
            {
                EditorGUILayout.LabelField("Fitted field","Multi-bounce diffuse radiance");
                if(GUILayout.Button("Show multi-bounce field"))
                {
                    if(replayRig==null)ApplyCurrent();
                    fieldQuantity=RadiosityQuantity;fieldMode=0;fieldExperiment=experiment;
                    showIrradiance=true;irradianceAdvanced=true;fieldDirty=true;RefreshEditorViews();
                }
            }
            LightSpec[] targets=TargetLights(report);
            int onlyLight=VisibleReplayLight(entry.lights.Length);
            for(int i=0;i<entry.lights.Length;i++)
            {
                if(onlyLight>=0&&i!=onlyLight)continue;
                LightSpec light=entry.lights[i];
                EditorGUILayout.LabelField(ReplayLightLabel(i),light.position.ToString("F4")+" / "+light.intensity.ToString("F4"));
                LightSpec target=i<targets.Length?targets[i]:null;
                if(target==null)continue;
                EditorGUILayout.LabelField("Position error",Vector3.Distance(light.position,target.position).ToString("F4")+" m");
                if(report.usesCct)EditorGUILayout.LabelField("CCT",light.cctKelvin.ToString("F0")+" K (target "+target.cctKelvin.ToString("F0")+" K)");
                if(light.kind==LightKind.Spot)EditorGUILayout.LabelField("Direction error",Vector3.Angle(light.direction,target.direction).ToString("F4")+" deg");
            }
            EditorGUILayout.Space();
            EditorGUILayout.HelpBox("在 Game 视图观察照明变化；Scene 视图中黄色球为目标位置，青色球为当前灯位置。记录包含每个 Adam 步骤和每个被接受的 L-BFGS 步骤。",MessageType.None);
            if(GUILayout.Button("Open Game View"))EditorApplication.ExecuteMenuItem("Window/General/Game");
        }

        private void DrawReplayView(int lightCount)
        {
            EditorGUILayout.LabelField("Recorded lights",lightCount.ToString());
            if(lightCount<2)return;
            bool isolate=EditorGUILayout.Popup("Replay view",isolateReplayLight?1:0,replayViews)==1;
            int selected=Mathf.Clamp(selectedReplayLight,0,lightCount-1);
            if(isolate)
            {
                var lightLabels=new string[lightCount];
                for(int i=0;i<lightCount;i++)lightLabels[i]=ReplayLightLabel(i);
                selected=EditorGUILayout.Popup("Selected light",selected,lightLabels);
            }
            if(isolate!=isolateReplayLight||isolate&&selected!=selectedReplayLight)SetReplayView(isolate,selected);
        }

        private void OpenExperimentScene()
        {
            StopReplay();DestroyFieldOverlays();fieldDirty=true;
            if(SceneManager.GetActiveScene().path!=ScenePath)
            {
                if(!Application.isBatchMode&&!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())return;
                EditorSceneManager.OpenScene(ScenePath);
            }
            index=0;ApplyCurrent();
        }

        private void StopReplay()
        {
            playing=false;
            if(replayRig==null)return;
            replayRig.Dispose();replayRig=null;fieldDirty=true;RefreshEditorViews();
        }

        private void ApplyCurrent()
        {
            var report=reports?[experiment];
            if(SceneManager.GetActiveScene().path!=ScenePath||RecordedLightCount(report)==0){StopReplay();return;}
            var history=report.history;
            index=Mathf.Clamp(index,0,history.Count-1);
            var lights=history[index].lights;if(lights==null||lights.Length==0){StopReplay();return;}
            if(replayRig==null)replayRig=new CornellSceneRig(SceneManager.GetActiveScene());
            if(replayRig.Count<lights.Length){StopReplay();Debug.LogWarning("The saved scene is missing candidate lights.");return;}
            replayRig.Apply(lights,report.usesVisibility,VisibleReplayLight(lights.Length));fieldDirty=true;
            RefreshEditorViews();Repaint();
        }

        private void Tick()
        {
            if(SceneManager.GetActiveScene().path!=ScenePath)
            {
                StopReplay();
                if(fieldSamples!=null){DestroyFieldOverlays();fieldDirty=true;}
            }
            if(showIrradiance&&autoFieldUpdate&&EditorApplication.timeSinceStartup>=nextFieldUpdate){nextFieldUpdate=EditorApplication.timeSinceStartup+0.25;UpdateFieldIfLightsChanged();}
            if(!playing||candidate==null||reports==null||reports[experiment]==null)return;
            if(EditorApplication.timeSinceStartup<nextUpdate)return;
            var history=reports[experiment].history;
            if(index>=history.Count-1){playing=false;Repaint();return;}
            index++;nextUpdate=EditorApplication.timeSinceStartup+1.0/speed;ApplyCurrent();
        }

        private void DrawPerturbationTab()
        {
            EditorGUILayout.HelpBox("对当前场景中位于 Bounding Box 内的 Point / Spot / Area 灯进行可复现随机扰动。方向灯不包含在内。",MessageType.Info);
            if(!boundsInitialized){FitBoundsToLights();boundsInitialized=true;}
            EditorGUILayout.HelpBox("在 Scene 视图中编辑生效范围：拖动中心操作杆移动 Box；拖动六个面的控制点缩放 Box。",MessageType.Info);
            selectedLightsOnly=EditorGUILayout.Toggle("Selected objects only",selectedLightsOnly);
            EditorGUILayout.BeginHorizontal();
            if(GUILayout.Button("Fit bounds to lights"))FitBoundsToLights();
            if(GUILayout.Button("Select lights in box"))Selection.objects=GetLightsInScope().ConvertAll(l=>(UnityEngine.Object)l.gameObject).ToArray();
            EditorGUILayout.EndHorizontal();
            Light[] scoped=GetLightsInScope().ToArray();
            EditorGUILayout.LabelField("Lights in scope",scoped.Length.ToString());
            EditorGUILayout.Space();
            seed=EditorGUILayout.IntField("Random seed",seed);
            positionSigma=EditorGUILayout.FloatField("Position sigma (m)",Mathf.Max(0,positionSigma));
            intensitySigma=EditorGUILayout.FloatField("Log-intensity sigma",Mathf.Max(0,intensitySigma));
            directionSigma=EditorGUILayout.FloatField("Direction sigma (degrees)",Mathf.Max(0,directionSigma));
            colorSigma=EditorGUILayout.FloatField("Linear color sigma",Mathf.Max(0,colorSigma));
            EditorGUILayout.Space();
            EditorGUI.BeginDisabledGroup(scoped.Length==0);
            if(GUILayout.Button("Apply random perturbation"))ApplyRandomPerturbation();
            EditorGUI.EndDisabledGroup();
            EditorGUI.BeginDisabledGroup(perturbationBaseline.Count==0);
            if(GUILayout.Button("Restore pre-perturbation lights"))RestorePerturbationBaseline();
            EditorGUI.EndDisabledGroup();
            if(lastPerturbedCount>=0)EditorGUILayout.LabelField("Last perturbed lights",lastPerturbedCount.ToString());
            EditorGUILayout.HelpBox("每次应用后随机种子自动加一。相同种子与初始状态会得到相同扰动。操作支持 Undo；Restore 会恢复本窗口第一次扰动前的灯光。",MessageType.None);
        }

        private System.Collections.Generic.List<Light> GetPerturbableLights(bool ignoreBounds)
        {
            var result=new System.Collections.Generic.List<Light>();
            foreach(var light in UnityEngine.Object.FindObjectsOfType<Light>())
            {
                if(light.type!=LightType.Point&&light.type!=LightType.Spot&&light.type!=LightType.Rectangle&&light.type!=LightType.Disc)continue;
                if(selectedLightsOnly&&!System.Array.Exists(Selection.gameObjects,go=>go==light.gameObject))continue;
                if(!ignoreBounds&&!effectBounds.Contains(light.transform.position))continue;
                result.Add(light);
            }
            return result;
        }

        private System.Collections.Generic.List<Light> GetLightsInScope() { return GetPerturbableLights(false); }

        private void FitBoundsToLights()
        {
            var lights=GetPerturbableLights(true);if(lights.Count==0)return;
            Bounds bounds=new Bounds(lights[0].transform.position,Vector3.zero);
            foreach(var light in lights)bounds.Encapsulate(light.transform.position);
            bounds.Expand(new Vector3(0.5f,0.5f,0.5f));effectBounds=bounds;SceneView.RepaintAll();
        }

        private void ApplyRandomPerturbation()
        {
            var lights=GetLightsInScope();if(lights.Count==0)return;
            foreach(var light in lights)if(!perturbationBaseline.Exists(s=>s.light==light))perturbationBaseline.Add(CaptureLightState(light));
            var undoObjects=new System.Collections.Generic.List<UnityEngine.Object>();foreach(var light in lights){undoObjects.Add(light);undoObjects.Add(light.transform);}
            Undo.RecordObjects(undoObjects.ToArray(),"Random lighting perturbation");
            var random=new System.Random(seed);
            foreach(var light in lights)
            {
                Vector3 offset=new Vector3(Gaussian(random),Gaussian(random),Gaussian(random))*positionSigma;
                light.transform.position+=offset;
                light.intensity=Mathf.Max(0,light.intensity*Mathf.Exp(intensitySigma*Gaussian(random)));
                if(light.type!=LightType.Point)
                {
                    float angle=Mathf.Tan(directionSigma*Mathf.Deg2Rad);
                    Vector3 direction=light.transform.forward+angle*new Vector3(Gaussian(random),Gaussian(random),Gaussian(random));
                    if(direction.sqrMagnitude>1e-8f)light.transform.rotation=Quaternion.LookRotation(direction.normalized,light.transform.up);
                }
                if(colorSigma>0)
                {
                    Color c=light.color;
                    light.color=new Color(Mathf.Clamp01(c.r+colorSigma*Gaussian(random)),Mathf.Clamp01(c.g+colorSigma*Gaussian(random)),Mathf.Clamp01(c.b+colorSigma*Gaussian(random)),c.a);
                }
                EditorUtility.SetDirty(light);
            }
            lastPerturbedCount=lights.Count;Debug.Log("Perturbed "+lights.Count+" lights with seed "+seed);seed++;
            SceneView.RepaintAll();EditorApplication.QueuePlayerLoopUpdate();
        }

        private static LightState CaptureLightState(Light light)
        {
            return new LightState { light=light,position=light.transform.position,rotation=light.transform.rotation,intensity=light.intensity,color=light.color };
        }

        private void RestorePerturbationBaseline()
        {
            foreach(var state in perturbationBaseline)
            {
                if(state.light==null)continue;Undo.RecordObjects(new UnityEngine.Object[]{state.light,state.light.transform},"Restore lighting perturbation");
                state.light.transform.position=state.position;state.light.transform.rotation=state.rotation;
                state.light.intensity=state.intensity;state.light.color=state.color;EditorUtility.SetDirty(state.light);
            }
            perturbationBaseline.Clear();lastPerturbedCount=-1;SceneView.RepaintAll();EditorApplication.QueuePlayerLoopUpdate();
        }

        private static float Gaussian(System.Random random)
        {
            double u=1-random.NextDouble(),v=random.NextDouble();
            return (float)(Math.Sqrt(-2*Math.Log(Math.Max(u,1e-12)))*Math.Cos(2*Math.PI*v));
        }

        private void DrawPerturbationHandles(SceneView view)
        {
            Matrix4x4 previous=Handles.matrix;Handles.matrix=Matrix4x4.identity;
            boundsHandle.handleColor=new Color(0.1f,0.85f,1,1);boundsHandle.wireframeColor=new Color(0.1f,0.85f,1,1);
            boundsHandle.center=effectBounds.center;boundsHandle.size=effectBounds.size;
            EditorGUI.BeginChangeCheck();boundsHandle.DrawHandle();
            if(EditorGUI.EndChangeCheck()){effectBounds=new Bounds(boundsHandle.center,boundsHandle.size);Repaint();}
            Handles.color=new Color(0.1f,0.85f,1,1);Handles.DrawWireCube(effectBounds.center,effectBounds.size);
            Handles.Label(effectBounds.center+Vector3.up*(effectBounds.size.y*0.5f+HandleUtility.GetHandleSize(effectBounds.center)*0.12f),"Light perturbation scope");
            Handles.matrix=previous;
            foreach(var light in GetPerturbableLights(true))
            {
                bool inside=effectBounds.Contains(light.transform.position);
                Handles.color=inside?new Color(0.2f,1,0.35f):new Color(0.5f,0.5f,0.5f,0.5f);
                float size=HandleUtility.GetHandleSize(light.transform.position)*0.06f;
                Handles.SphereHandleCap(0,light.transform.position,Quaternion.identity,size,EventType.Repaint);
                if(inside)Handles.Label(light.transform.position+Vector3.up*size,light.name);
            }
            Repaint();
        }
        private void DrawIrradianceControls()
        {
            if(fieldOverlayMaterial!=null)fieldOverlayMaterial.SetFloat("_Alpha",fieldOverlayOpacity);
            if(fieldDirty)RecomputeIrradiance();
            irradianceAdvanced=EditorGUILayout.Foldout(irradianceAdvanced,"Advanced Irradiance Settings",true);
            if(!irradianceAdvanced)return;
            EditorGUI.indentLevel++;
            EditorGUI.BeginChangeCheck();
            fieldMode=EditorGUILayout.Popup("Field",fieldMode,fieldModes);
            fieldQuantity=EditorGUILayout.Popup("Quantity",fieldQuantity,new[]{"Direct irradiance","Direct outgoing radiance","Multi-bounce outgoing radiance"});
            if(RadiosityField)EditorGUILayout.LabelField("Visualization","Solver patches (Scene view)");
            else fieldDisplayMode=EditorGUILayout.Popup("Visualization",fieldDisplayMode,new[]{"Continuous surface","Sample discs (debug)"});
            int onlyLight=IsolatedPreviewLight();
            if(onlyLight>=0)EditorGUILayout.LabelField("Replay scope",ReplayLightLabel(onlyLight)+" only");
            if(fieldMode>0)
            {
                if(FieldUsesReplayTarget(onlyLight))EditorGUILayout.LabelField("Target rig","Current replay experiment");
                else
                {
                    fieldExperiment=EditorGUILayout.Popup("Target rig",fieldExperiment,labels);
                    if(reports[fieldExperiment]==null)EditorGUILayout.HelpBox("Run the selected Cornell experiment to create its target field.",MessageType.Info);
                }
            }
            if(!RadiosityField)fieldResolution=EditorGUILayout.IntSlider("Surface grid resolution",fieldResolution,4,50);
            rgbFieldMode=EditorGUILayout.Toggle("RGB instead of heatmap",rgbFieldMode);
            if(!rgbFieldMode)fieldColormap=EditorGUILayout.Popup("Heatmap palette",fieldColormap,new[]{"Viridis","Blue-Cyan-Green-Yellow-Red"});
            autoFieldNormalize=EditorGUILayout.Toggle("Auto normalize",autoFieldNormalize);
            if(!autoFieldNormalize)manualFieldMax=EditorGUILayout.FloatField("Maximum field value",Mathf.Max(1e-8f,manualFieldMax));
            logFieldScale=EditorGUILayout.Toggle("Log brightness",logFieldScale);
            if(logFieldScale)fieldContrast=EditorGUILayout.Slider("Log contrast",fieldContrast,1,100);
            if(!RadiosityField&&fieldDisplayMode==0)fieldOverlayOpacity=EditorGUILayout.Slider("Surface overlay opacity",fieldOverlayOpacity,0.05f,1f);
            else fieldMarkerScale=EditorGUILayout.Slider("Marker size",fieldMarkerScale,0.2f,2f);
            autoFieldUpdate=EditorGUILayout.Toggle("Refresh when lights change",autoFieldUpdate);
            if(EditorGUI.EndChangeCheck()){fieldDirty=true;if(!rgbFieldMode)BuildFieldLegend();}
            EditorGUILayout.BeginHorizontal();
            if(GUILayout.Button("Evaluate now"))fieldDirty=true;
            if(GUILayout.Button("Export displayed CSV"))ExportDisplayedField();
            EditorGUILayout.EndHorizontal();
            if(fieldValues!=null)
            {
                EditorGUILayout.LabelField(!RadiosityField&&fieldDisplayMode==0?"Surface vertices":"Surface samples",fieldSamples.Count.ToString());
                EditorGUILayout.LabelField("Maximum component",fieldMax.ToString("E6"));
                if(!rgbFieldMode)
                {
                    if(fieldLegend==null)BuildFieldLegend();
                    Rect rect=GUILayoutUtility.GetRect(100,18,GUILayout.ExpandWidth(true));GUI.DrawTexture(rect,fieldLegend,ScaleMode.StretchToFill);
                    EditorGUILayout.LabelField("0",GetFieldDisplayMax().ToString("E3")+" (normalized scale)");
                }
            }
            EditorGUI.indentLevel--;
        }
        private void BuildVisualizationGeometry(MeshRenderer[] renderers,int resolution)
        {
            DestroyFieldOverlays();
            var positions=new System.Collections.Generic.List<Vector3>();var normals=new System.Collections.Generic.List<Vector3>();
            var areas=new System.Collections.Generic.List<float>();var albedos=new System.Collections.Generic.List<Color>();var names=new System.Collections.Generic.List<string>();
            if(fieldDisplayMode==0)
            {
                fieldOverlayRoot=new GameObject("Irradiance Continuous Preview") { hideFlags=HideFlags.HideAndDontSave };
                fieldOverlayMaterial=new Material(Shader.Find("LightingMvp/IrradianceOverlay")) { hideFlags=HideFlags.HideAndDontSave };
                fieldOverlayMaterial.SetFloat("_Alpha",fieldOverlayOpacity);
            }
            foreach(var renderer in renderers)
            {
                var filter=renderer.GetComponent<MeshFilter>();Mesh source=filter!=null?filter.sharedMesh:null;if(source==null)continue;
                Vector3[] vertices=source.vertices,sourceNormals=source.normals;
                int[] indices=source.triangles;
                Matrix4x4 normalMatrix=renderer.transform.worldToLocalMatrix.transpose;
                Color albedo=SurfaceSampleSet.DiffuseAlbedo(renderer);
                int start=positions.Count;
                var overlayVertices=new System.Collections.Generic.List<Vector3>();
                var triangles=new System.Collections.Generic.List<int>();
                for(int t=0;t<indices.Length;t+=3)
                {
                    int ia=indices[t],ib=indices[t+1],ic=indices[t+2];
                    Vector3 a=renderer.transform.TransformPoint(vertices[ia]),b=renderer.transform.TransformPoint(vertices[ib]),c=renderer.transform.TransformPoint(vertices[ic]);
                    Vector3 geometricNormal=Vector3.Cross(b-a,c-a);
                    if(geometricNormal.sqrMagnitude<1e-20f)continue;
                    bool hasNormals=sourceNormals.Length==vertices.Length;
                    Vector3 na=hasNormals?normalMatrix.MultiplyVector(sourceNormals[ia]):geometricNormal;
                    Vector3 nb=hasNormals?normalMatrix.MultiplyVector(sourceNormals[ib]):geometricNormal;
                    Vector3 nc=hasNormals?normalMatrix.MultiplyVector(sourceNormals[ic]):geometricNormal;
                    float area=geometricNormal.magnitude/(6f*resolution*resolution);
                    int first=overlayVertices.Count;
                    for(int i=0;i<=resolution;i++)for(int j=0;j<=resolution-i;j++)
                    {
                        float u=i/(float)resolution,v=j/(float)resolution;
                        Vector3 pos=a+u*(b-a)+v*(c-a),normal=(na+u*(nb-na)+v*(nc-na)).normalized;
                        positions.Add(pos);normals.Add(normal);areas.Add(0);albedos.Add(albedo);names.Add(renderer.name);
                        overlayVertices.Add(pos+normal*0.003f);
                        if(i+j==resolution)continue;
                        int current=first+i*(resolution+1)-i*(i-1)/2+j;
                        int next=current+resolution+1-i;
                        AddFace(current,next,current+1);
                        if(i+j<resolution-1)AddFace(current+1,next,next+1);
                    }

                    void AddFace(int i,int j,int k)
                    {
                        triangles.Add(i);triangles.Add(j);triangles.Add(k);
                    }
                    for(int n=triangles.Count-3*resolution*resolution;n<triangles.Count;n++)areas[start+triangles[n]]+=area;
                }
                if(fieldDisplayMode==0&&overlayVertices.Count>0)
                {
                    var mesh=new Mesh { name=renderer.name+" Irradiance Grid",hideFlags=HideFlags.HideAndDontSave };
                    mesh.SetVertices(overlayVertices);mesh.SetTriangles(triangles,0);mesh.RecalculateBounds();
                    var overlay=new GameObject(renderer.name+" Irradiance") { hideFlags=HideFlags.HideAndDontSave };
                    overlay.transform.SetParent(fieldOverlayRoot.transform,false);overlay.AddComponent<MeshFilter>().sharedMesh=mesh;
                    var meshRenderer=overlay.AddComponent<MeshRenderer>();meshRenderer.sharedMaterial=fieldOverlayMaterial;
                    meshRenderer.shadowCastingMode=UnityEngine.Rendering.ShadowCastingMode.Off;meshRenderer.receiveShadows=false;
                    meshRenderer.lightProbeUsage=UnityEngine.Rendering.LightProbeUsage.Off;meshRenderer.reflectionProbeUsage=UnityEngine.Rendering.ReflectionProbeUsage.Off;
                    fieldOverlays.Add(new FieldOverlay { mesh=mesh,start=start,count=overlayVertices.Count });
                }
            }
            fieldSamples=new SurfaceSampleSet { positions=positions.ToArray(),normals=normals.ToArray(),areas=areas.ToArray(),albedos=albedos.ToArray(),names=names.ToArray() };
        }

        private void UpdateFieldOverlayColors()
        {
            if(fieldValues==null)return;
            foreach(var overlay in fieldOverlays)
            {
                var colors=new Color[overlay.count];
                for(int i=0;i<overlay.count;i++)colors[i]=FieldColor(fieldValues[overlay.start+i]);
                overlay.mesh.colors=colors;
            }
        }

        private void DestroyFieldOverlays()
        {
            foreach(var overlay in fieldOverlays)if(overlay.mesh!=null)UnityEngine.Object.DestroyImmediate(overlay.mesh);
            fieldOverlays.Clear();fieldSamples=null;fieldValues=null;fieldTransport=null;fieldGeometrySignature="";
            if(fieldOverlayRoot!=null)UnityEngine.Object.DestroyImmediate(fieldOverlayRoot);fieldOverlayRoot=null;
            if(fieldOverlayMaterial!=null)UnityEngine.Object.DestroyImmediate(fieldOverlayMaterial);fieldOverlayMaterial=null;
        }

        private LightSpec[] CurrentFieldLights(int onlyLight)
        {
            if(onlyLight>=0)
            {
                Light light=replayRig[onlyLight];
                return light!=null&&light.isActiveAndEnabled?new[]{UnityLightAdapter.FromUnity(light)}:Array.Empty<LightSpec>();
            }
            var specs=new System.Collections.Generic.List<LightSpec>();
            Scene scene=SceneManager.GetActiveScene();
            foreach(var light in UnityEngine.Object.FindObjectsOfType<Light>())
                if(light.gameObject.scene==scene&&light.isActiveAndEnabled&&(light.type==LightType.Point||light.type==LightType.Spot))specs.Add(UnityLightAdapter.FromUnity(light));
            return specs.ToArray();
        }

        private bool FieldUsesReplayTarget(int onlyLight)=>onlyLight>=0||
            (RadiosityField&&replayRig!=null&&reports[experiment].usesRadiosity);

        private void RecomputeIrradiance()
        {
            fieldDirty=false;if(SceneManager.GetActiveScene().path!=ScenePath)return;
            var root=GameObject.Find("Cornell Box - 2m open front");if(root==null)return;
            var renderers=root.GetComponentsInChildren<MeshRenderer>();
            string geometry=GeometrySignature(renderers);
            if(geometry!=fieldGeometrySignature||fieldSamples==null)
            {
                if(RadiosityField)
                {
                    DestroyFieldOverlays();
                    fieldSamples=SurfaceSampleSet.FromRenderers(renderers,CornellBoxMvpEditor.RadiosityResolution);
                    fieldTransport=new DiffuseRadiosity(fieldSamples);
                }
                else BuildVisualizationGeometry(renderers,fieldResolution);
                fieldGeometrySignature=geometry;
            }
            if(fieldOverlayMaterial!=null)fieldOverlayMaterial.SetFloat("_Alpha",fieldOverlayOpacity);
            int onlyLight=IsolatedPreviewLight();
            var targetReport=fieldMode>0?reports[FieldUsesReplayTarget(onlyLight)?experiment:fieldExperiment]:null;
            LightSpec[] targetSpecs=targetReport!=null?TargetLights(targetReport):null;
            if(fieldMode>0&&(targetSpecs==null||onlyLight>=0&&(onlyLight>=targetSpecs.Length||targetSpecs[onlyLight]==null)))
            {
                DestroyFieldOverlays();RefreshEditorViews();return;
            }
            if(targetSpecs!=null&&onlyLight>=0)targetSpecs=new[]{targetSpecs[onlyLight]};
            bool shadows=RadiosityField||(targetReport?.usesVisibility??(replayRig==null||reports[experiment].usesVisibility));
            float[] Mask(LightSpec[] specs)=>shadows?UnityVisibility.Evaluate(specs,fieldSamples):null;
            if(RadiosityField)Plot((specs,cct)=>AnalyticLighting.EvaluateField(specs,fieldSamples,Mask(specs),cct,fieldTransport));
            else
            {
                var shader=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/LightingMvp/Shaders/SurfaceLighting.compute");
                using var gpu=new SurfaceLightingGpu(shader);
                Plot((specs,cct)=>
                {
                    gpu.Evaluate(specs,fieldSamples,out Vector3[] irradiance,out Vector3[] radiance,Mask(specs),cct);
                    return fieldQuantity==0?irradiance:radiance;
                });
            }
            fieldMax=0;foreach(var value in fieldValues)fieldMax=Mathf.Max(fieldMax,Mathf.Max(value.x,Mathf.Max(value.y,value.z)));
            UpdateFieldOverlayColors();
            fieldLightSignature=CurrentLightSignature();RefreshEditorViews();

            void Plot(Func<LightSpec[],bool,Vector3[]> Evaluate)
            {
                if(fieldMode==0)fieldValues=Evaluate(CurrentFieldLights(onlyLight),false);
                else
                {
                    Vector3[] target=Evaluate(targetSpecs,targetReport.usesCct);
                    if(fieldMode==1)fieldValues=target;
                    else
                    {
                        Vector3[] current=Evaluate(CurrentFieldLights(onlyLight),false);
                        fieldValues=new Vector3[current.Length];
                        for(int i=0;i<fieldValues.Length;i++)fieldValues[i]=new Vector3(Mathf.Abs(current[i].x-target[i].x),Mathf.Abs(current[i].y-target[i].y),Mathf.Abs(current[i].z-target[i].z));
                    }
                }
            }
        }

        private string CurrentLightSignature()
        {
            var text=new System.Text.StringBuilder();text.Append(SceneManager.GetActiveScene().path);
            var root=GameObject.Find("Cornell Box - 2m open front");
            if(root!=null)text.Append(GeometrySignature(root.GetComponentsInChildren<MeshRenderer>()));
            foreach(var light in UnityEngine.Object.FindObjectsOfType<Light>())
            {
                if(!light.isActiveAndEnabled||(light.type!=LightType.Point&&light.type!=LightType.Spot))continue;
                text.Append(light.GetInstanceID()).Append(light.transform.position.ToString("R")).Append(light.transform.forward.ToString("R"))
                    .Append(light.intensity).Append(light.range).Append(light.color.ToString("R")).Append(light.spotAngle).Append(light.innerSpotAngle);
            }
            return text.ToString();
        }

        private string GeometrySignature(MeshRenderer[] renderers)
        {
            var text=new System.Text.StringBuilder();
            if(RadiosityField)
            {
                text.Append("radiosity");
                foreach(var collider in FindObjectsOfType<Collider>())
                {
                    if(!collider.enabled||collider.isTrigger)continue;
                    text.Append('/').Append(collider.GetInstanceID()).Append('/').Append(collider.bounds.min.ToString("R"))
                        .Append('/').Append(collider.bounds.max.ToString("R")).Append('/').Append(collider.transform.localToWorldMatrix.ToString("R"));
                }
            }
            else text.Append(fieldResolution).Append('/').Append(fieldDisplayMode);
            foreach(var renderer in renderers)
            {
                var mesh=renderer.GetComponent<MeshFilter>()?.sharedMesh;
                text.Append('/').Append(renderer.GetInstanceID()).Append('/').Append(mesh!=null?mesh.GetInstanceID():0)
                    .Append('/').Append(renderer.transform.localToWorldMatrix.ToString("R")).Append('/').Append(SurfaceSampleSet.DiffuseAlbedo(renderer).ToString("R"));
            }
            return text.ToString();
        }

        private void UpdateFieldIfLightsChanged()
        {
            string signature=CurrentLightSignature();if(signature!=fieldLightSignature){fieldLightSignature=signature;fieldDirty=true;}
            if(fieldOverlayMaterial!=null)fieldOverlayMaterial.SetFloat("_Alpha",fieldOverlayOpacity);
            if(fieldDirty)RecomputeIrradiance();
        }

        private float GetFieldDisplayMax() { return autoFieldNormalize?Mathf.Max(fieldMax,1e-12f):Mathf.Max(manualFieldMax,1e-12f); }
        private static float Luminance(Vector3 value) { return 0.2126f*value.x+0.7152f*value.y+0.0722f*value.z; }
        private float FieldTransfer(float value)
        {
            float t=Mathf.Clamp01(value/GetFieldDisplayMax());
            return logFieldScale?Mathf.Log(1+t*fieldContrast)/Mathf.Log(1+fieldContrast):t;
        }
        private Color FieldColor(Vector3 value)
        {
            if(rgbFieldMode)return new Color(FieldTransfer(value.x),FieldTransfer(value.y),FieldTransfer(value.z),1);
            return Heatmap(FieldTransfer(Luminance(value)),fieldColormap);
        }
        private static Color Heatmap(float t,int palette)
        {
            if(palette==1)return Color.HSVToRGB(Mathf.Lerp(0.66f,0,Mathf.Clamp01(t)),1,1);
            Color[] colors={ new Color(0.267f,0.005f,0.329f),new Color(0.229f,0.322f,0.545f),new Color(0.128f,0.567f,0.551f),new Color(0.369f,0.789f,0.383f),new Color(0.993f,0.906f,0.144f) };
            float scaled=Mathf.Clamp01(t)*(colors.Length-1);int index=Mathf.Min(colors.Length-2,Mathf.FloorToInt(scaled));
            return Color.Lerp(colors[index],colors[index+1],scaled-index);
        }
        private void BuildFieldLegend()
        {
            if(fieldLegend!=null)UnityEngine.Object.DestroyImmediate(fieldLegend);
            fieldLegend=new Texture2D(256,8,TextureFormat.RGBA32,false) { hideFlags=HideFlags.HideAndDontSave };
            for(int x=0;x<256;x++)for(int y=0;y<8;y++)fieldLegend.SetPixel(x,y,Heatmap(x/255f,fieldColormap));
            fieldLegend.Apply();
        }
        private void DrawIrradianceMarkers(SceneView view)
        {
            if((!RadiosityField&&fieldDisplayMode==0)||fieldSamples==null||fieldValues==null)return;
            for(int i=0;i<fieldSamples.Count;i++)
            {
                Handles.color=FieldColor(fieldValues[i]);
                float radius=Mathf.Sqrt(fieldSamples.areas[i])*0.38f*fieldMarkerScale;
                Handles.DrawSolidDisc(fieldSamples.positions[i]+fieldSamples.normals[i]*0.002f,fieldSamples.normals[i],radius);
            }
        }
        private void ExportDisplayedField()
        {
            if(fieldValues==null||fieldSamples==null)return;
            string quantity=RadiosityField?"radiosity":fieldQuantity==0?"irradiance":"radiance";
            Directory.CreateDirectory(OutputRoot);string path=Path.Combine(OutputRoot,"debug_"+quantity+"_view.csv");
            using(var writer=new StreamWriter(path))
            {
                writer.WriteLine($"surface,x,y,z,nx,ny,nz,area,{quantity}_r,{quantity}_g,{quantity}_b");
                for(int i=0;i<fieldSamples.Count;i++)writer.WriteLine(string.Format(System.Globalization.CultureInfo.InvariantCulture,
                    "{0},{1:R},{2:R},{3:R},{4:R},{5:R},{6:R},{7:R},{8:R},{9:R},{10:R}",fieldSamples.names[i],
                    fieldSamples.positions[i].x,fieldSamples.positions[i].y,fieldSamples.positions[i].z,fieldSamples.normals[i].x,fieldSamples.normals[i].y,fieldSamples.normals[i].z,
                    fieldSamples.areas[i],fieldValues[i].x,fieldValues[i].y,fieldValues[i].z));
            }
            Debug.Log("Irradiance view exported: "+path);
        }
        private static void RefreshEditorViews() { SceneView.RepaintAll();EditorApplication.QueuePlayerLoopUpdate();UnityEditorInternal.InternalEditorUtility.RepaintAllViews(); }

        private void DrawSceneMarkers(SceneView view)
        {
            if(activeTab==1)DrawPerturbationHandles(view);
            if(showIrradiance)DrawIrradianceMarkers(view);
            if(activeTab!=0||replayRig==null||reports[experiment]==null||SceneManager.GetActiveScene().path!=ScenePath)return;
            var report=reports[experiment];if(report.history==null||report.history.Count==0)return;
            LightSpec[] current=report.history[Mathf.Clamp(index,0,report.history.Count-1)].lights;
            LightSpec[] targets=TargetLights(report);
            int onlyLight=VisibleReplayLight(current.Length);
            for(int i=0;i<current.Length;i++)
            {
                if(onlyLight>=0&&i!=onlyLight)continue;
                Vector3 position=current[i].position;
                Handles.color=new Color(0.1f,0.9f,1,1);
                Handles.SphereHandleCap(0,position,Quaternion.identity,0.08f,EventType.Repaint);
                Handles.Label(position+Vector3.up*0.10f,"Current "+(i+1));
                if(!showTarget||i>=targets.Length||targets[i]==null)continue;
                Vector3 target=targets[i].position;
                Handles.color=Color.yellow;
                Handles.SphereHandleCap(0,target,Quaternion.identity,0.10f,EventType.Repaint);
                Handles.Label(target+Vector3.up*0.14f,"Target "+(i+1));
                Handles.DrawDottedLine(position,target,4);
            }
        }
    }
}
