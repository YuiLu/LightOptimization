using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

using UnityEngine.SceneManagement;

namespace LightingMvp.Editor
{
    public static class CornellBoxMvpEditor
    {
        private const string ScenePath="Assets/Scenes/CornellBoxMvp.unity";
        private const string ComputePath="Assets/LightingMvp/Shaders/SurfaceLighting.compute";
        private static string OutputRoot { get { return Path.GetFullPath(Path.Combine(Application.dataPath,"../../results/unity_mvp")); } }

        [Serializable] public sealed class GradientReport
        {
            public string name; public float maxAbsoluteError,relativeNormError,cosine; public bool passed;
        }
        [Serializable] public sealed class ExperimentReport
        {
            public string name,optimizer; public int steps,lbfgsEvaluations; public bool success; public float initialLoss,finalLoss,initialNrmse,finalNrmse;
            public float positionError,intensityRelativeError,directionErrorDegrees;
            public LightSpec target,initial,optimized;
            public List<OptimizationHistoryEntry> history;
        }
        [Serializable] public sealed class MvpReport
        {
            public string backend="LightOpt analytic CPU + Unity Compute Shader GPU";
            public int sampleCount; public float gpuMaxAbsoluteError,gpuMaxRelativeError,gpuRadianceMaxAbsoluteError;
            public GradientReport[] gradients; public ExperimentReport[] experiments;
        }
        [MenuItem("Lighting MVP/Run Cornell Box Experiments")]
        public static void RunAll()
        {
            Directory.CreateDirectory(OutputRoot);
            EnsureExperimentScene();
            SurfaceSampleSet samples=LoadSceneSamples(20);
            LightSpec pointTarget=PointTarget(),pointInitial=PointInitial();
            LightSpec spotTarget=SpotTarget(),spotInitial=SpotInitial();
            var report=new MvpReport { sampleCount=samples.Count };
            ValidateGpu(pointTarget,samples,report);
            ValidateGpu(spotTarget,samples,report);
            report.gradients=new[] { CheckGradient("point",pointInitial,samples),CheckGradient("spot",spotInitial,samples) };
            report.experiments=new[]
            {
                RunExperiment("point_position_intensity",pointTarget,pointInitial,samples,500),
                RunExperiment("spot_position_direction",spotTarget,spotInitial,samples,700),
            };
            File.WriteAllText(Path.Combine(OutputRoot,"report.json"),JsonUtility.ToJson(report,true));
            ApplyLight(pointTarget);
            EditorSceneManager.SaveScene(SceneManager.GetActiveScene(),ScenePath);
            Debug.Log("Lighting MVP complete: "+Path.Combine(OutputRoot,"report.json"));
            foreach(var e in report.experiments)
                Debug.Log(string.Format("{0}: NRMSE {1:P2} -> {2:P4}, position error {3:F5} m, direction error {4:F3} deg",
                    e.name,e.initialNrmse,e.finalNrmse,e.positionError,e.directionErrorDegrees));
            AssetDatabase.Refresh();
            if(Array.Exists(report.gradients,g=>!g.passed)||Array.Exists(report.experiments,e=>!e.success)) throw new Exception("Lighting MVP validation failed; inspect report.json");
        }

        private static void EnsureExperimentScene() { if(SceneManager.GetActiveScene().path==ScenePath)return; string fullPath=Path.Combine(Application.dataPath,"../",ScenePath); if(!File.Exists(fullPath))throw new FileNotFoundException("The saved Cornell Box experiment asset is required",fullPath); EditorSceneManager.OpenScene(ScenePath); }

        private static SurfaceSampleSet LoadSceneSamples(int resolution) { var root=GameObject.Find("Cornell Box - 2m open front"); if(root==null)throw new InvalidOperationException("Cornell Box experiment root is missing from the saved scene"); return SurfaceSampleSet.FromRenderers(root.GetComponentsInChildren<MeshRenderer>(),resolution); }

        public static void BatchRun()
        {
            try { RunAll(); EditorApplication.Exit(0); }
            catch(Exception e) { Debug.LogException(e); EditorApplication.Exit(2); }
        }

        private static ExperimentReport RunExperiment(string name,LightSpec target,LightSpec initial,SurfaceSampleSet samples,int steps)
        {
            LightSpec[] targetRig={target},initialRig={initial};
            Vector3[] targetField=AnalyticLighting.Evaluate(targetRig,samples);
            Vector3[] initialField=AnalyticLighting.Evaluate(initialRig,samples);
            var options=new OptimizerOptions { steps=steps,lbfgsIterations=200,position=true,intensity=name.StartsWith("point"),direction=name.StartsWith("spot"),color=false };
            OptimizationResult result=LightRigOptimizer.Optimize(initialRig,targetField,samples,options);
            LightSpec optimized=result.optimized[0];
            var report=new ExperimentReport { name=name,optimizer="Adam + L-BFGS",steps=steps,lbfgsEvaluations=result.lbfgsEvaluations,success=result.finalNrmse<0.02f,target=target.Clone(),initial=initial.Clone(),optimized=optimized.Clone(),
                initialLoss=result.initialLoss,finalLoss=result.finalLoss,initialNrmse=result.initialNrmse,finalNrmse=result.finalNrmse,
                positionError=Vector3.Distance(target.position,optimized.position),
                intensityRelativeError=Mathf.Abs(target.intensity-optimized.intensity)/target.intensity,
                directionErrorDegrees=target.kind==LightKind.Spot?Vector3.Angle(target.direction,optimized.direction):0,
                history=result.history };
            File.WriteAllText(Path.Combine(OutputRoot,name+".json"),JsonUtility.ToJson(report,true));
            using(var writer=new StreamWriter(Path.Combine(OutputRoot,name+"_history.csv")))
            {
                writer.WriteLine("step,loss,nrmse,position_x,position_y,position_z,intensity,direction_x,direction_y,direction_z");
                foreach(var h in result.history) { LightSpec light=h.lights[0]; writer.WriteLine(string.Format(CultureInfo.InvariantCulture,"{0},{1:R},{2:R},{3:R},{4:R},{5:R},{6:R},{7:R},{8:R},{9:R}",h.step,h.loss,h.nrmse,light.position.x,light.position.y,light.position.z,light.intensity,light.direction.x,light.direction.y,light.direction.z)); }
            }
            SaveField(Path.Combine(OutputRoot,name+"_field.csv"),samples,targetField,initialField,
                AnalyticLighting.Evaluate(result.optimized,samples));
            ApplyLight(target); Capture(name+"_target.png");
            ApplyLight(initial); Capture(name+"_initial.png");
            ApplyLight(optimized); Capture(name+"_optimized.png");
            return report;
        }

        private static GradientReport CheckGradient(string name,LightSpec spec,SurfaceSampleSet samples)
        {
            LightSpec[] rig={spec}; float[] x=AnalyticLighting.ToVector(rig),analytic=new float[x.Length];
            Vector3[] target=AnalyticLighting.Evaluate(new[]{PointTarget()},samples);
            AnalyticLighting.LossAndGradient(x,rig,target,samples,true,true,true,true,analytic);
            var finite=new float[x.Length]; const float h=1e-3f;
            for(int i=0;i<x.Length;i++)
            {
                if(spec.kind==LightKind.Point && i>=4 && i<=6) continue;
                float original=x[i];x[i]=original+h;float plus=AnalyticLighting.Loss(x,rig,target,samples);
                x[i]=original-h;float minus=AnalyticLighting.Loss(x,rig,target,samples);x[i]=original;
                finite[i]=(plus-minus)/(2*h);
            }
            double aa=0,ff=0,af=0,diff=0,max=0;
            for(int i=0;i<x.Length;i++)
            {
                aa+=analytic[i]*analytic[i];ff+=finite[i]*finite[i];af+=analytic[i]*finite[i];
                diff+=(analytic[i]-finite[i])*(analytic[i]-finite[i]);max=Math.Max(max,Math.Abs(analytic[i]-finite[i]));
            }
            float cosine=(float)(af/Math.Max(Math.Sqrt(aa*ff),1e-20));
            float relative=(float)(Math.Sqrt(diff)/Math.Max(Math.Sqrt(aa),Math.Sqrt(ff)));
            return new GradientReport { name=name,maxAbsoluteError=(float)max,relativeNormError=relative,cosine=cosine,
                passed=cosine>0.9999f && relative<0.01f };
        }

        private static void ValidateGpu(LightSpec spec,SurfaceSampleSet samples,MvpReport report)
        {
            var shader=AssetDatabase.LoadAssetAtPath<ComputeShader>(ComputePath);
            using(var gpu=new SurfaceLightingGpu(shader))
            {
                gpu.Evaluate(new[]{spec},samples,out Vector3[] e,out Vector3[] r);
                Vector3[] cpuE=AnalyticLighting.Evaluate(new[]{spec},samples),cpuR=AnalyticLighting.DiffuseRadiance(cpuE,samples);
                for(int j=0;j<e.Length;j++)
                {
                    report.gpuMaxAbsoluteError=Mathf.Max(report.gpuMaxAbsoluteError,(e[j]-cpuE[j]).magnitude);
                    report.gpuMaxRelativeError=Mathf.Max(report.gpuMaxRelativeError,(e[j]-cpuE[j]).magnitude/Mathf.Max(cpuE[j].magnitude,1e-5f));
                    report.gpuRadianceMaxAbsoluteError=Mathf.Max(report.gpuRadianceMaxAbsoluteError,(r[j]-cpuR[j]).magnitude);
                }
                using(var writer=new StreamWriter(Path.Combine(OutputRoot,"gpu_surface_field_"+spec.kind.ToString().ToLowerInvariant()+".csv")))
                {
                    writer.WriteLine("surface,x,y,z,nx,ny,nz,area,irradiance_r,irradiance_g,irradiance_b,radiance_r,radiance_g,radiance_b");
                    for(int j=0;j<e.Length;j++) writer.WriteLine(string.Format(CultureInfo.InvariantCulture,
                        "{0},{1:R},{2:R},{3:R},{4:R},{5:R},{6:R},{7:R},{8:R},{9:R},{10:R},{11:R},{12:R},{13:R}",
                        samples.names[j],samples.positions[j].x,samples.positions[j].y,samples.positions[j].z,
                        samples.normals[j].x,samples.normals[j].y,samples.normals[j].z,samples.areas[j],
                        e[j].x,e[j].y,e[j].z,r[j].x,r[j].y,r[j].z));
                }
            }
            if(report.gpuMaxAbsoluteError>2e-4f) throw new Exception("CPU/GPU irradiance mismatch: "+report.gpuMaxAbsoluteError);
        }

        private static void SaveField(string path,SurfaceSampleSet s,Vector3[] target,Vector3[] initial,Vector3[] optimized)
        {
            using(var writer=new StreamWriter(path))
            {
                writer.WriteLine("surface,x,y,z,nx,ny,nz,area,target_r,target_g,target_b,initial_r,initial_g,initial_b,optimized_r,optimized_g,optimized_b");
                for(int j=0;j<s.Count;j++) writer.WriteLine(string.Format(CultureInfo.InvariantCulture,
                    "{0},{1:R},{2:R},{3:R},{4:R},{5:R},{6:R},{7:R},{8:R},{9:R},{10:R},{11:R},{12:R},{13:R},{14:R},{15:R},{16:R}",
                    s.names[j],s.positions[j].x,s.positions[j].y,s.positions[j].z,s.normals[j].x,s.normals[j].y,s.normals[j].z,s.areas[j],
                    target[j].x,target[j].y,target[j].z,initial[j].x,initial[j].y,initial[j].z,optimized[j].x,optimized[j].y,optimized[j].z));
            }
        }

        private static LightSpec PointTarget()
        {
            return new LightSpec { position=new Vector3(0.25f,1.35f,0.10f),intensity=2.5f,color=new Color(1,0.85f,0.70f,1) };
        }
        private static LightSpec PointInitial()
        {
            return new LightSpec { position=new Vector3(-0.30f,0.95f,-0.25f),intensity=1.25f,color=new Color(1,0.85f,0.70f,1) };
        }
        private static LightSpec SpotTarget()
        {
            return new LightSpec { position=new Vector3(-0.45f,1.65f,-0.45f),intensity=4,color=new Color(1,0.9f,0.8f,1),
                kind=LightKind.Spot,direction=new Vector3(0.2f,-1,0.5f).normalized,range=8,innerAngle=25,outerAngle=50 };
        }
        private static LightSpec SpotInitial()
        {
            return new LightSpec { position=new Vector3(0.10f,1.20f,-0.10f),intensity=4,color=new Color(1,0.9f,0.8f,1),
                kind=LightKind.Spot,direction=new Vector3(0.65f,-0.55f,-0.30f).normalized,range=8,innerAngle=25,outerAngle=50 };
        }

        private static void ApplyLight(LightSpec spec) { UnityLightAdapter.Apply(GameObject.Find("Candidate Light").GetComponent<Light>(),spec); }

        private static void Capture(string filename)
        {
            var camera=GameObject.Find("Validation Camera").GetComponent<Camera>();
            var rt=new RenderTexture(768,768,24,RenderTextureFormat.ARGB32,RenderTextureReadWrite.sRGB);
            var previous=RenderTexture.active;camera.targetTexture=rt;camera.Render();
            RenderTexture.active=rt;var texture=new Texture2D(768,768,TextureFormat.RGB24,false);
            texture.ReadPixels(new Rect(0,0,768,768),0,0);texture.Apply();
            File.WriteAllBytes(Path.Combine(OutputRoot,filename),texture.EncodeToPNG());
            camera.targetTexture=null;RenderTexture.active=previous;
            UnityEngine.Object.DestroyImmediate(texture);rt.Release();UnityEngine.Object.DestroyImmediate(rt);
        }

    }
}





