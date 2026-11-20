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
        [SerializeField] private int fieldResolution=30,fieldMode,fieldExperiment,fieldDisplayMode,fieldColormap=1;
        [SerializeField] private bool showIrradiance,irradianceAdvanced,autoFieldUpdate=true,autoFieldNormalize=true,logFieldScale=true,rgbFieldMode;
        [SerializeField] private float fieldContrast=20,fieldMarkerScale=1,manualFieldMax=0.0005f,fieldOverlayOpacity=1;
        private bool fieldDirty=true;
        private double nextFieldUpdate;
        private string fieldLightSignature="";
        private SurfaceSampleSet fieldSamples;
        private Vector3[] fieldValues;
        private float fieldMax;
        private Texture2D fieldLegend;
        private readonly System.Collections.Generic.List<FieldOverlay> fieldOverlays=new System.Collections.Generic.List<FieldOverlay>();
        private GameObject fieldOverlayRoot;
        private Material fieldOverlayMaterial;
        private sealed class FieldOverlay { public Mesh mesh;public int start,count; }
        private readonly string[] fieldModes={ "Current scene lights","Target irradiance","Absolute error |current-target|" };
        private readonly BoxBoundsHandle boundsHandle=new BoxBoundsHandle();
        private readonly System.Collections.Generic.List<LightState> perturbationBaseline=new System.Collections.Generic.List<LightState>();
        private int lastPerturbedCount=-1;

        private sealed class LightState { public Light light;public Vector3 position;public Quaternion rotation;public float intensity;public Color color; }

        private const string ScenePath="Assets/Scenes/CornellBoxMvp.unity";
        private static string OutputRoot { get { return Path.GetFullPath(Path.Combine(Application.dataPath,"../../results/unity_mvp")); } }
        private readonly string[] files={ "point_position_intensity.json","spot_position_direction.json" };
        private readonly string[] labels={ "Point: position + intensity","Spot: position + direction" };
        private CornellBoxMvpEditor.ExperimentReport[] reports;
        private int experiment,index;
        private float speed=45;
        private bool playing,showTarget=true;
        private double nextUpdate;
        private Light candidate;
        private LightSpec restoreOnClose;

        [MenuItem("Lighting MVP/Debug Window")]
        public static void Open()
        {
            var window=GetWindow<LightingDebugWindow>();window.titleContent=new GUIContent("Lighting Debug");window.Show();
        }

        public static void BatchPreviewSmoke() { try { var window=GetWindow<LightingDebugWindow>();window.OpenExperimentScene();for(int experimentIndex=0;experimentIndex<2;experimentIndex++){window.experiment=experimentIndex;var history=window.reports[experimentIndex].history;window.index=Mathf.Min(250,history.Count-1);window.ApplyCurrent();Vector3 expected=history[window.index].lights[0].position;if(Vector3.Distance(window.candidate.transform.position,expected)>1e-6f)throw new Exception("Iteration preview state mismatch");}window.Close();Debug.Log("Iteration preview smoke test passed");EditorApplication.Exit(0); } catch(Exception e){Debug.LogException(e);EditorApplication.Exit(2);} }

        public static void BatchPerturbationSmoke() { try { var window=GetWindow<LightingDebugWindow>();window.OpenExperimentScene();window.activeTab=1;window.effectBounds=new Bounds(new Vector3(0,1,0),new Vector3(2,2,2));if(window.GetLightsInScope().Count!=1)throw new Exception("Bounding-box scope mismatch");window.seed=1234;Vector3 before=window.candidate.transform.position;window.ApplyRandomPerturbation();Vector3 perturbed=window.candidate.transform.position;if(Vector3.Distance(before,perturbed)<1e-6f)throw new Exception("Perturbation did not move the light");window.RestorePerturbationBaseline();if(Vector3.Distance(before,window.candidate.transform.position)>1e-6f)throw new Exception("Perturbation restore mismatch");window.seed=1234;window.ApplyRandomPerturbation();if(Vector3.Distance(perturbed,window.candidate.transform.position)>1e-6f)throw new Exception("Perturbation is not reproducible");window.RestorePerturbationBaseline();window.effectBounds=new Bounds(new Vector3(10,10,10),Vector3.one);if(window.GetLightsInScope().Count!=0)throw new Exception("Bounding-box exclusion mismatch");window.Close();Debug.Log("Lighting perturbation smoke test passed");EditorApplication.Exit(0); } catch(Exception e){Debug.LogException(e);EditorApplication.Exit(2);} }

        public static void BatchIrradianceSmoke() { try { var window=GetWindow<LightingDebugWindow>();window.OpenExperimentScene();window.showIrradiance=true;window.fieldMode=0;window.RecomputeIrradiance();if(window.fieldSamples.Count!=5*(window.fieldResolution+1)*(window.fieldResolution+1)||window.fieldValues.Length!=window.fieldSamples.Count||window.fieldMax<=0)throw new Exception("Current irradiance visualization data mismatch");if(window.fieldDisplayMode==0&&(window.fieldOverlays.Count!=5||window.fieldOverlays[0].mesh.colors.Length==0))throw new Exception("Continuous irradiance mesh was not generated");window.fieldMode=1;window.fieldExperiment=0;window.RecomputeIrradiance();if(window.fieldMax<=0)throw new Exception("Target irradiance visualization data mismatch");window.fieldMode=2;window.RecomputeIrradiance();if(window.fieldMax<=0)throw new Exception("Error irradiance visualization data mismatch");window.ExportDisplayedField();window.Close();Debug.Log("Irradiance visualization smoke test passed");EditorApplication.Exit(0); } catch(Exception e){Debug.LogException(e);EditorApplication.Exit(2);} }

        public static void OpenContinuousIrradianceVisual() { var window=GetWindow<LightingDebugWindow>();window.OpenExperimentScene();window.showIrradiance=true;window.fieldDisplayMode=0;window.fieldMode=1;window.fieldExperiment=0;window.RecomputeIrradiance();EditorApplication.ExecuteMenuItem("Window/General/Game"); }

        private void OnEnable()
        {
            reports=new CornellBoxMvpEditor.ExperimentReport[files.Length];
            for(int i=0;i<files.Length;i++)
            {
                string path=Path.Combine(OutputRoot,files[i]);
                if(File.Exists(path))reports[i]=JsonUtility.FromJson<CornellBoxMvpEditor.ExperimentReport>(File.ReadAllText(path));
            }
            BuildFieldLegend();DestroyFieldOverlays();fieldDirty=true;
            EditorApplication.update+=Tick;
            SceneView.duringSceneGui+=DrawSceneMarkers;
        }

        private void OnDisable()
        {
            EditorApplication.update-=Tick;SceneView.duringSceneGui-=DrawSceneMarkers;
            if(fieldLegend!=null)UnityEngine.Object.DestroyImmediate(fieldLegend);DestroyFieldOverlays();
            if(candidate!=null&&restoreOnClose!=null){UnityLightAdapter.Apply(candidate,restoreOnClose);EditorUtility.SetDirty(candidate);}
        }

        private void OnGUI()
        {
            activeTab=Mathf.Clamp(activeTab,0,1);activeTab=GUILayout.Toolbar(activeTab,new[]{"Iteration Replay","Light Perturbation"});
            EditorGUILayout.Space();bool enabled=EditorGUILayout.ToggleLeft("Show Irradiance",showIrradiance,EditorStyles.boldLabel);
            if(enabled!=showIrradiance){showIrradiance=enabled;fieldDirty=true;if(!showIrradiance){DestroyFieldOverlays();fieldValues=null;}RefreshEditorViews();}
            if(showIrradiance&&SceneManager.GetActiveScene().path!=ScenePath){EditorGUILayout.HelpBox("Irradiance preview currently uses the saved Cornell Box experiment.",MessageType.Info);if(GUILayout.Button("Open Cornell Box Experiment"))OpenExperimentScene();}
            else if(showIrradiance)DrawIrradianceControls();
            EditorGUILayout.Space();if(activeTab==0)DrawIterationTab();else DrawPerturbationTab();
        }

        private void DrawIterationTab()
        {
            if(reports==null||reports[0]==null||reports[1]==null)
            {
                EditorGUILayout.HelpBox("先运行 Lighting MVP > Run Cornell Box Experiments 生成逐迭代记录。",MessageType.Info);return;
            }
            if(SceneManager.GetActiveScene().path!=ScenePath)
            {
                EditorGUILayout.HelpBox("迭代预览使用已保存的 Cornell Box 实验场景。",MessageType.Info);
                if(GUILayout.Button("Open Cornell Box Experiment"))OpenExperimentScene();
                return;
            }
            BindCandidate();
            EditorGUI.BeginChangeCheck();
            experiment=EditorGUILayout.Popup("Experiment",experiment,labels);
            if(EditorGUI.EndChangeCheck()){index=0;playing=false;ApplyCurrent();}
            var report=reports[experiment];
            if(report.history==null||report.history.Count==0||report.history[0].lights==null)
            {
                EditorGUILayout.HelpBox("旧报告没有灯具快照，请重新运行实验。",MessageType.Warning);return;
            }
            int count=report.history.Count;
            EditorGUI.BeginChangeCheck();
            index=EditorGUILayout.IntSlider("Recorded state",index,0,count-1);
            if(EditorGUI.EndChangeCheck()){playing=false;ApplyCurrent();}
            EditorGUILayout.BeginHorizontal();
            if(GUILayout.Button("|<")){index=0;playing=false;ApplyCurrent();}
            if(GUILayout.Button("<")){index=Mathf.Max(0,index-1);playing=false;ApplyCurrent();}
            playing=GUILayout.Toggle(playing,playing?"Pause":"Play","Button",GUILayout.Width(80));
            if(GUILayout.Button(">")){index=Mathf.Min(count-1,index+1);playing=false;ApplyCurrent();}
            if(GUILayout.Button(">|")){index=count-1;playing=false;ApplyCurrent();}
            EditorGUILayout.EndHorizontal();
            speed=EditorGUILayout.Slider("States / second",speed,1,120);
            showTarget=EditorGUILayout.Toggle("Show target marker",showTarget);
            EditorGUILayout.Space();
            OptimizationHistoryEntry entry=report.history[index];LightSpec light=entry.lights[0];
            string phase=entry.step<=report.steps?"Adam":"L-BFGS accepted step";
            EditorGUILayout.LabelField(phase,"Step "+entry.step);
            EditorGUILayout.LabelField("Loss",entry.loss.ToString("E6"));
            EditorGUILayout.LabelField("Field NRMSE",entry.nrmse.ToString("P6"));
            EditorGUILayout.LabelField("Position",light.position.ToString("F6"));
            EditorGUILayout.LabelField("Intensity",light.intensity.ToString("F6"));
            if(light.kind==LightKind.Spot)EditorGUILayout.LabelField("Direction",light.direction.normalized.ToString("F6"));
            EditorGUILayout.LabelField("Position error",Vector3.Distance(light.position,report.target.position).ToString("F6")+" m");
            if(light.kind==LightKind.Spot)EditorGUILayout.LabelField("Direction error",Vector3.Angle(light.direction,report.target.direction).ToString("F4")+" deg");
            EditorGUILayout.Space();
            EditorGUILayout.HelpBox("在 Game 视图观察照明变化；Scene 视图中黄色球为目标位置，青色球为当前灯位置。记录包含每个 Adam 步骤和每个被接受的 L-BFGS 步骤。",MessageType.None);
            if(GUILayout.Button("Open Game View"))EditorApplication.ExecuteMenuItem("Window/General/Game");
        }

        private void OpenExperimentScene()
        {
            DestroyFieldOverlays();EditorSceneManager.OpenScene(ScenePath);candidate=null;restoreOnClose=null;BindCandidate();index=0;ApplyCurrent();
        }

        private void BindCandidate()
        {
            if(candidate!=null)return;
            var go=GameObject.Find("Candidate Light");if(go==null)return;
            candidate=go.GetComponent<Light>();restoreOnClose=UnityLightAdapter.FromUnity(candidate);ApplyCurrent();
        }

        private void ApplyCurrent()
        {
            if(candidate==null||reports==null||reports[experiment]==null)return;
            var history=reports[experiment].history;if(history==null||history.Count==0)return;
            index=Mathf.Clamp(index,0,history.Count-1);
            var lights=history[index].lights;if(lights==null||lights.Length==0)return;
            UnityLightAdapter.Apply(candidate,lights[0]);EditorUtility.SetDirty(candidate);
            SceneView.RepaintAll();EditorApplication.QueuePlayerLoopUpdate();Repaint();
        }

        private void Tick()
        {
            if(showIrradiance&&autoFieldUpdate&&EditorApplication.timeSinceStartup>=nextFieldUpdate){nextFieldUpdate=EditorApplication.timeSinceStartup+0.25;UpdateFieldIfLightsChanged();}
            if(!playing||candidate==null||reports==null||reports[experiment]==null)return;
            if(EditorApplication.timeSinceStartup<nextUpdate)return;
            var history=reports[experiment].history;
            if(index>=history.Count-1){playing=false;return;}
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
            fieldDisplayMode=EditorGUILayout.Popup("Visualization",fieldDisplayMode,new[]{"Continuous surface","Sample discs (debug)"});
            fieldMode=EditorGUILayout.Popup("Field",fieldMode,fieldModes);
            if(fieldMode>0)fieldExperiment=EditorGUILayout.Popup("Target rig",fieldExperiment,labels);
            fieldResolution=EditorGUILayout.IntSlider("Surface grid resolution",fieldResolution,4,50);
            rgbFieldMode=EditorGUILayout.Toggle("RGB instead of heatmap",rgbFieldMode);
            if(!rgbFieldMode)fieldColormap=EditorGUILayout.Popup("Heatmap palette",fieldColormap,new[]{"Viridis","Blue-Cyan-Green-Yellow-Red"});
            autoFieldNormalize=EditorGUILayout.Toggle("Auto normalize",autoFieldNormalize);
            if(!autoFieldNormalize)manualFieldMax=EditorGUILayout.FloatField("Maximum irradiance",Mathf.Max(1e-8f,manualFieldMax));
            logFieldScale=EditorGUILayout.Toggle("Log brightness",logFieldScale);
            if(logFieldScale)fieldContrast=EditorGUILayout.Slider("Log contrast",fieldContrast,1,100);
            if(fieldDisplayMode==0)fieldOverlayOpacity=EditorGUILayout.Slider("Surface overlay opacity",fieldOverlayOpacity,0.05f,1f);
            else fieldMarkerScale=EditorGUILayout.Slider("Marker size",fieldMarkerScale,0.2f,2f);
            autoFieldUpdate=EditorGUILayout.Toggle("Refresh when lights change",autoFieldUpdate);
            if(EditorGUI.EndChangeCheck()){fieldDirty=true;if(!rgbFieldMode)BuildFieldLegend();}
            EditorGUILayout.BeginHorizontal();
            if(GUILayout.Button("Evaluate now"))fieldDirty=true;
            if(GUILayout.Button("Export displayed CSV"))ExportDisplayedField();
            EditorGUILayout.EndHorizontal();
            if(fieldValues!=null)
            {
                EditorGUILayout.LabelField(fieldDisplayMode==0?"Surface vertices":"Surface samples",fieldSamples.Count.ToString());
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
        private void BuildVisualizationGeometry(System.Collections.Generic.IEnumerable<MeshRenderer> renderers,int resolution)
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
                if(source.vertexCount!=4||source.triangles.Length!=6)throw new System.NotSupportedException("Continuous preview currently expects quad surfaces: "+renderer.name);
                Vector3[] vertices=source.vertices;Vector3 origin=renderer.transform.TransformPoint(vertices[0]);
                Vector3 u=renderer.transform.TransformPoint(vertices[1])-origin,v=renderer.transform.TransformPoint(vertices[3])-origin;
                Vector3 normal=renderer.transform.TransformDirection(source.normals[0]).normalized;
                Material material=renderer.sharedMaterial;Color albedo=material!=null&&material.HasProperty("_BaseColor")?material.GetColor("_BaseColor"):Color.white;
                float area=Vector3.Cross(u,v).magnitude/(resolution*resolution);
                var surfaceVertices=new System.Collections.Generic.List<Vector3>();var surfaceNormals=new System.Collections.Generic.List<Vector3>();
                var surfaceUvs=new System.Collections.Generic.List<Vector2>();var triangles=new System.Collections.Generic.List<int>();
                int start=positions.Count;
                for(int i=0;i<=resolution;i++)for(int j=0;j<=resolution;j++)
                {
                    Vector3 position=origin+(i/(float)resolution)*u+(j/(float)resolution)*v;
                    positions.Add(position);normals.Add(normal);areas.Add(area);albedos.Add(albedo);names.Add(renderer.name);
                    surfaceVertices.Add(position+normal*0.003f);surfaceNormals.Add(normal);surfaceUvs.Add(new Vector2(i/(float)resolution,j/(float)resolution));
                    if(i<resolution&&j<resolution)
                    {
                        int index=i*(resolution+1)+j;
                        triangles.Add(index);triangles.Add(index+resolution+1);triangles.Add(index+1);
                        triangles.Add(index+1);triangles.Add(index+resolution+1);triangles.Add(index+resolution+2);
                    }
                }
                if(fieldDisplayMode==0)
                {
                    var mesh=new Mesh { name=renderer.name+" Irradiance Grid",hideFlags=HideFlags.HideAndDontSave };
                    mesh.SetVertices(surfaceVertices);mesh.SetNormals(surfaceNormals);mesh.SetUVs(0,surfaceUvs);mesh.SetTriangles(triangles,0);mesh.RecalculateBounds();
                    var overlay=new GameObject(renderer.name+" Irradiance") { hideFlags=HideFlags.HideAndDontSave };
                    overlay.transform.SetParent(fieldOverlayRoot.transform,false);overlay.AddComponent<MeshFilter>().sharedMesh=mesh;
                    var meshRenderer=overlay.AddComponent<MeshRenderer>();meshRenderer.sharedMaterial=fieldOverlayMaterial;
                    meshRenderer.shadowCastingMode=UnityEngine.Rendering.ShadowCastingMode.Off;meshRenderer.receiveShadows=false;
                    meshRenderer.lightProbeUsage=UnityEngine.Rendering.LightProbeUsage.Off;meshRenderer.reflectionProbeUsage=UnityEngine.Rendering.ReflectionProbeUsage.Off;
                    fieldOverlays.Add(new FieldOverlay { mesh=mesh,start=start,count=surfaceVertices.Count });
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
            fieldOverlays.Clear();
            if(fieldOverlayRoot!=null)UnityEngine.Object.DestroyImmediate(fieldOverlayRoot);fieldOverlayRoot=null;
            if(fieldOverlayMaterial!=null)UnityEngine.Object.DestroyImmediate(fieldOverlayMaterial);fieldOverlayMaterial=null;
            foreach(var go in Resources.FindObjectsOfTypeAll<GameObject>())if(go.name=="Irradiance Continuous Preview")UnityEngine.Object.DestroyImmediate(go);
        }
        private void RecomputeIrradiance()
        {
            fieldDirty=false;if(SceneManager.GetActiveScene().path!=ScenePath)return;
            var root=GameObject.Find("Cornell Box - 2m open front");if(root==null)return;
            BuildVisualizationGeometry(root.GetComponentsInChildren<MeshRenderer>(),fieldResolution);
            var currentSpecs=new System.Collections.Generic.List<LightSpec>();
            foreach(var light in UnityEngine.Object.FindObjectsOfType<Light>())
                if(light.type==LightType.Point||light.type==LightType.Spot)currentSpecs.Add(UnityLightAdapter.FromUnity(light));
            if(currentSpecs.Count==0)currentSpecs.Add(new LightSpec { intensity=0 });
            LightSpec[] targetSpecs={ reports[fieldExperiment].target };
            var shader=AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/LightingMvp/Shaders/SurfaceLighting.compute");
            using(var gpu=new SurfaceLightingGpu(shader))
            {
                gpu.Evaluate(currentSpecs.ToArray(),fieldSamples,out Vector3[] current,out Vector3[] _);
                if(fieldMode==0){fieldValues=current;}
                else
                {
                    gpu.Evaluate(targetSpecs,fieldSamples,out Vector3[] target,out Vector3[] _);
                    if(fieldMode==1)fieldValues=target;
                    else
                    {
                        fieldValues=new Vector3[current.Length];
                        for(int i=0;i<fieldValues.Length;i++)fieldValues[i]=new Vector3(Mathf.Abs(current[i].x-target[i].x),Mathf.Abs(current[i].y-target[i].y),Mathf.Abs(current[i].z-target[i].z));
                    }
                }
            }
            fieldMax=0;foreach(var value in fieldValues)fieldMax=Mathf.Max(fieldMax,Mathf.Max(value.x,Mathf.Max(value.y,value.z)));
            UpdateFieldOverlayColors();
            fieldLightSignature=CurrentLightSignature();RefreshEditorViews();
        }

        private string CurrentLightSignature()
        {
            var text=new System.Text.StringBuilder();text.Append(SceneManager.GetActiveScene().path);
            foreach(var light in UnityEngine.Object.FindObjectsOfType<Light>())
            {
                if(light.type!=LightType.Point&&light.type!=LightType.Spot)continue;
                text.Append(light.GetInstanceID()).Append(light.transform.position).Append(light.transform.forward)
                    .Append(light.intensity).Append(light.range).Append(light.color).Append(light.spotAngle).Append(light.innerSpotAngle);
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
            if(fieldLegend!=null)UnityEngine.Object.DestroyImmediate(fieldLegend);DestroyFieldOverlays();
            fieldLegend=new Texture2D(256,8,TextureFormat.RGBA32,false) { hideFlags=HideFlags.HideAndDontSave };
            for(int x=0;x<256;x++)for(int y=0;y<8;y++)fieldLegend.SetPixel(x,y,Heatmap(x/255f,fieldColormap));
            fieldLegend.Apply();
        }
        private void DrawIrradianceMarkers(SceneView view)
        {
            if(fieldDisplayMode==0||fieldSamples==null||fieldValues==null)return;
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
            Directory.CreateDirectory(OutputRoot);string path=Path.Combine(OutputRoot,"debug_irradiance_view.csv");
            using(var writer=new StreamWriter(path))
            {
                writer.WriteLine("surface,x,y,z,nx,ny,nz,area,irradiance_r,irradiance_g,irradiance_b");
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
            if(activeTab!=0||reports==null||reports[experiment]==null||SceneManager.GetActiveScene().path!=ScenePath)return;
            var report=reports[experiment];if(report.history==null||report.history.Count==0)return;
            LightSpec current=report.history[Mathf.Clamp(index,0,report.history.Count-1)].lights[0];
            Handles.color=new Color(0.1f,0.9f,1,1);
            Handles.SphereHandleCap(0,current.position,Quaternion.identity,0.10f,EventType.Repaint);
            Handles.Label(current.position+Vector3.up*0.10f,"Current");
            if(showTarget)
            {
                Handles.color=Color.yellow;
                Handles.SphereHandleCap(0,report.target.position,Quaternion.identity,0.12f,EventType.Repaint);
                Handles.Label(report.target.position+Vector3.up*0.14f,"Target");
                Handles.DrawDottedLine(current.position,report.target.position,4);
            }
        }
    }
}



























