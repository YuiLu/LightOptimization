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
        internal const string ScenePath="Assets/Scenes/CornellBoxMvp.unity";
        internal const string RadiosityExperiment="three_point_radiosity_cct";
        internal const int RadiosityResolution=4;
        private const string ComputePath="Assets/LightingMvp/Shaders/SurfaceLighting.compute";
        internal static string OutputRoot { get { return Path.GetFullPath(Path.Combine(Application.dataPath,"../../results/unity_mvp")); } }

        [Serializable] public sealed class GradientReport
        { public string name;public float maxAbsoluteError,relativeNormError,cosine;public bool passed,allMasksFixed=true; }
        [Serializable] public sealed class LightError
        {
            public int targetIndex,optimizedIndex;
            public float positionError,intensityRelativeError,cctErrorKelvin,directionErrorDegrees;
        }
        [Serializable] public sealed class ExperimentReport
        {
            public string name,optimizer;public int steps,lbfgsEvaluations;public bool success,fieldPassed,parameterPassed,usesVisibility,usesCct,usesRadiosity;
            public float initialLoss,finalLoss,initialNrmse,finalNrmse,positionError,intensityRelativeError,directionErrorDegrees;
            public float maxPositionError,permutationPositionError,cctErrorKelvin,cctRelativeError,visibilityMismatch;
            public int[] parameterPermutation;
            public LightError[] labelErrors,alignedErrors;
            public float alignedVisibilityMismatch,alignedIntensityRelativeError,alignedCctErrorKelvin;
            public int blockedTargetRays,blockedInitialRays,blockedFinalRays;
            public LightSpec target,initial,optimized;
            public LightSpec[] targetRig,initialRig,optimizedRig;
            public List<OptimizationHistoryEntry> history;
        }
        [Serializable] public sealed class MvpReport
        {
            public string backend="Multi-light analytic CPU + Unity Compute Shader direct light + CPU diffuse radiosity + detached Unity collider visibility";
            public int sampleCount,multiLightSampleCount;public float gpuMaxAbsoluteError,gpuMaxRelativeError,gpuRadianceMaxAbsoluteError;
            public int blockedValidationRays;
            public float shadowFieldNrmse;
            public RadiosityReport radiosity;
            public GradientReport[] gradients;public ExperimentReport[] experiments;
        }

        [Serializable] public sealed class RadiosityReport
        {
            public int patches,blockedPairs,refinedPairs;
            public float maxRawRowSum,maxRowSum,maxReciprocityError,fixedPointNrmse;
            public float zeroBounceNrmse,oneBounceNrmse,fourBounceNrmse,sixteenBounceNrmse;
        }

        public static void RunAll()
        {
            if(EditorApplication.isPlayingOrWillChangePlaymode)throw new InvalidOperationException("Exit Play Mode before running Cornell validation.");
            LightingDebugWindow.EndCornellPreviews();
            if(!EnsureExperimentScene())return;
            using(var sceneRig=new CornellSceneRig(SceneManager.GetActiveScene()))
            {
                if(sceneRig.Count!=3)throw new InvalidOperationException("Update the saved Cornell scene with its three candidate lights before running.");
                Directory.CreateDirectory(OutputRoot);
                SurfaceSampleSet samples=LoadSceneSamples(10),multiSamples=LoadSceneSamples(6),radiositySamples=LoadSceneSamples(RadiosityResolution);
                LightSpec pointTarget=PointTarget(),pointInitial=PointInitial(),spotTarget=SpotTarget(),spotInitial=SpotInitial();
                LightSpec[] multiTarget=MultiTarget(),multiInitial=MultiInitial(false),multiCctInitial=MultiInitial(true);
                var report=new MvpReport { sampleCount=samples.Count,multiLightSampleCount=multiSamples.Count };
                ValidateGpu("point",new[]{pointTarget},samples,report);
                ValidateGpu("spot",new[]{spotTarget},samples,report);
                ValidateGpu("three_point_cct_shadow",multiTarget,multiSamples,report,true);
                var transport=new DiffuseRadiosity(radiositySamples);
                report.radiosity=CheckRadiosity(transport,multiTarget,radiositySamples);
                report.gradients=new[]
                {
                    CheckGradient("point",pointInitial,pointTarget,samples),CheckGradient("spot",spotInitial,spotTarget,samples),
                    CheckMultiGradient(multiTarget,multiCctInitial,multiSamples),CheckMultiGradient(multiTarget,multiCctInitial,radiositySamples,transport)
                };
                report.experiments=new[]
                {
                    RunExperiment("point_position_intensity",new[]{pointTarget},new[]{pointInitial},samples,500,false,false,false,sceneRig),
                    RunExperiment("spot_position_direction",new[]{spotTarget},new[]{spotInitial},samples,700,false,false,false,sceneRig),
                    RunExperiment("three_point_position_intensity",multiTarget,multiInitial,multiSamples,240,true,false,true,sceneRig),
                    RunExperiment("three_point_joint_cct",multiTarget,multiCctInitial,multiSamples,300,true,true,true,sceneRig),
                    RunExperiment(RadiosityExperiment,multiTarget,multiCctInitial,radiositySamples,300,true,true,true,sceneRig,transport),
                };
                File.WriteAllText(Path.Combine(OutputRoot,"report.json"),JsonUtility.ToJson(report,true));
                Debug.Log("Lighting MVP complete: "+Path.Combine(OutputRoot,"report.json"));
                foreach(var e in report.experiments)Debug.Log(string.Format("{0}: NRMSE {1:P2} -> {2:P4}, max position error {3:F5} m, mean CCT error {4:F2} K",e.name,e.initialNrmse,e.finalNrmse,e.maxPositionError,e.cctErrorKelvin));
                if(Array.Exists(report.gradients,g=>!g.passed)||Array.Exists(report.experiments,e=>!e.success))throw new Exception("Lighting MVP validation failed; inspect report.json");
            }
            LightingDebugWindow.RefreshCornellReports();
        }

        private static bool EnsureExperimentScene()
        {
            if(SceneManager.GetActiveScene().path==ScenePath)return true;
            if(AssetDatabase.LoadAssetAtPath<SceneAsset>(ScenePath)==null)throw new FileNotFoundException("The saved Cornell Box experiment asset is required",ScenePath);
            if(!Application.isBatchMode&&!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo())return false;
            EditorSceneManager.OpenScene(ScenePath);
            return true;
        }
        private static SurfaceSampleSet LoadSceneSamples(int resolution)
        {
            var root=GameObject.Find("Cornell Box - 2m open front");if(root==null)throw new InvalidOperationException("Cornell Box experiment root is missing");
            return SurfaceSampleSet.FromRenderers(root.GetComponentsInChildren<MeshRenderer>(),resolution);
        }
        public static void BatchRun(){try{RunAll();LightingDebugWindow.VerifyCornellPlayback();EditorApplication.Exit(0);}catch(Exception e){Debug.LogException(e);EditorApplication.Exit(2);}}

        private static ExperimentReport RunExperiment(string name,LightSpec[] target,LightSpec[] initial,SurfaceSampleSet samples,int steps,bool cctModel,bool optimizeColor,bool shadows,CornellSceneRig sceneRig,DiffuseRadiosity transport=null)
        {
            VisibilityProvider visibility=shadows?UnityVisibility.Evaluate:(VisibilityProvider)null;
            float[] targetMask=visibility?.Invoke(target,samples),initialMask=visibility?.Invoke(initial,samples);
            Vector3[] targetField=AnalyticLighting.EvaluateField(target,samples,targetMask,cctModel,transport);
            Vector3[] initialField=AnalyticLighting.EvaluateField(initial,samples,initialMask,cctModel,transport);
            var options=new OptimizerOptions { steps=steps,lbfgsIterations=150,position=true,intensity=!name.StartsWith("spot"),direction=name.StartsWith("spot"),color=optimizeColor,cct=cctModel };
            OptimizationResult result=LightRigOptimizer.Optimize(initial,targetField,samples,options,visibility,transport);
            LightSpec[] optimized=result.optimized;
            float positionError=0,maxPosition=0,intensityError=0,cctError=0,cctRelative=0;
            for(int i=0;i<target.Length;i++)
            {
                float dp=Vector3.Distance(target[i].position,optimized[i].position);positionError+=dp;maxPosition=Mathf.Max(maxPosition,dp);
                intensityError+=Mathf.Abs(target[i].intensity-optimized[i].intensity)/target[i].intensity;
                cctError+=Mathf.Abs(target[i].cctKelvin-optimized[i].cctKelvin);cctRelative+=Mathf.Abs(target[i].cctKelvin-optimized[i].cctKelvin)/target[i].cctKelvin;
            }
            int[] permutation=BestPermutation(target,optimized,out float permutationError);
            LightError[] alignedErrors=MeasureLights(target,optimized,permutation);
            float alignedIntensity=0,alignedCct=0;
            foreach(var e in alignedErrors){alignedIntensity+=e.intensityRelativeError;alignedCct+=e.cctErrorKelvin;}
            float[] finalMask=visibility?.Invoke(optimized,samples);
            float alignedVisibility=MaskMismatch(targetMask,finalMask,samples.Count,permutation);
            bool fieldPassed=result.finalNrmse<0.02f;
            bool parametersPassed=Array.TrueForAll(alignedErrors,e=>e.positionError<0.02f&&e.intensityRelativeError<0.05f&&
                e.directionErrorDegrees<2&&(!cctModel||e.cctErrorKelvin<200))&&(!shadows||alignedVisibility<0.01f);
            var report=new ExperimentReport
            {
                name=name,optimizer="Adam + L-BFGS",steps=steps,lbfgsEvaluations=result.lbfgsEvaluations,
                success=fieldPassed&&parametersPassed,fieldPassed=fieldPassed,parameterPassed=parametersPassed,
                usesVisibility=shadows,usesCct=cctModel,usesRadiosity=transport!=null,target=target[0].Clone(),initial=initial[0].Clone(),optimized=optimized[0].Clone(),
                targetRig=LightSpec.CloneArray(target),initialRig=LightSpec.CloneArray(initial),optimizedRig=LightSpec.CloneArray(optimized),
                initialLoss=result.initialLoss,finalLoss=result.finalLoss,initialNrmse=result.initialNrmse,finalNrmse=result.finalNrmse,
                positionError=positionError/target.Length,maxPositionError=maxPosition,permutationPositionError=permutationError,
                intensityRelativeError=intensityError/target.Length,cctErrorKelvin=cctError/target.Length,cctRelativeError=cctRelative/target.Length,
                directionErrorDegrees=target[0].kind==LightKind.Spot?Vector3.Angle(target[0].direction,optimized[0].direction):0,
                visibilityMismatch=MaskMismatch(targetMask,finalMask,samples.Count,Identity(target.Length)),
                alignedVisibilityMismatch=alignedVisibility,parameterPermutation=permutation,history=result.history,
                labelErrors=MeasureLights(target,optimized,Identity(target.Length)),alignedErrors=alignedErrors,
                alignedIntensityRelativeError=alignedIntensity/target.Length,alignedCctErrorKelvin=alignedCct/target.Length,
                blockedTargetRays=CountBlocked(targetMask),blockedInitialRays=CountBlocked(initialMask),blockedFinalRays=CountBlocked(finalMask),
            };
            File.WriteAllText(Path.Combine(OutputRoot,name+".json"),JsonUtility.ToJson(report,true));
            WriteHistory(name,result.history);SaveField(Path.Combine(OutputRoot,name+"_field.csv"),samples,targetField,initialField,
                AnalyticLighting.EvaluateField(optimized,samples,finalMask,cctModel,transport));
            sceneRig.Apply(target,shadows);Capture(name+"_target.png");sceneRig.Apply(initial,shadows);Capture(name+"_initial.png");sceneRig.Apply(optimized,shadows);Capture(name+"_optimized.png");
            return report;
        }

        private static LightError[] MeasureLights(LightSpec[] target,LightSpec[] result,int[] order)
        {
            var errors=new LightError[target.Length];
            for(int i=0;i<errors.Length;i++)
            {
                LightSpec a=target[i],b=result[order[i]];
                errors[i]=new LightError { targetIndex=i,optimizedIndex=order[i],positionError=Vector3.Distance(a.position,b.position),
                    intensityRelativeError=Mathf.Abs(a.intensity-b.intensity)/a.intensity,cctErrorKelvin=Mathf.Abs(a.cctKelvin-b.cctKelvin),
                    directionErrorDegrees=a.kind==LightKind.Spot?Vector3.Angle(a.direction,b.direction):0 };
            }
            return errors;
        }

        private static int CountBlocked(float[] mask)=>mask==null?0:Array.FindAll(mask,v=>v==0).Length;

        private static float MaskMismatch(float[] target,float[] result,int samples,int[] order)
        {
            if(target==null)return 0;
            int count=0;
            for(int i=0;i<order.Length;i++)for(int j=0;j<samples;j++)if(target[i*samples+j]!=result[order[i]*samples+j])count++;
            return count/(float)target.Length;
        }

        private static int[] BestPermutation(LightSpec[] target,LightSpec[] result,out float meanError)
        {
            if(target.Length!=3){meanError=0;for(int i=0;i<target.Length;i++)meanError+=Vector3.Distance(target[i].position,result[i].position)/target.Length;return Identity(target.Length);}
            int[][] permutations={new[]{0,1,2},new[]{0,2,1},new[]{1,0,2},new[]{1,2,0},new[]{2,0,1},new[]{2,1,0}};
            int[] best=permutations[0];float bestCost=float.MaxValue;
            foreach(var p in permutations){float cost=0;for(int i=0;i<3;i++)cost+=(target[i].position-result[p[i]].position).sqrMagnitude;if(cost<bestCost){bestCost=cost;best=p;}}
            meanError=0;for(int i=0;i<3;i++)meanError+=Vector3.Distance(target[i].position,result[best[i]].position)/3;return best;
        }
        private static int[] Identity(int count){var p=new int[count];for(int i=0;i<count;i++)p[i]=i;return p;}

        private static void WriteHistory(string name,List<OptimizationHistoryEntry> history)
        {
            using(var writer=new StreamWriter(Path.Combine(OutputRoot,name+"_history.csv")))
            {
                writer.WriteLine("step,light,loss,nrmse,position_x,position_y,position_z,intensity,cct_kelvin,direction_x,direction_y,direction_z");
                foreach(var h in history)for(int i=0;i<h.lights.Length;i++)
                {
                    LightSpec l=h.lights[i];writer.WriteLine(string.Format(CultureInfo.InvariantCulture,"{0},{1},{2:R},{3:R},{4:R},{5:R},{6:R},{7:R},{8:R},{9:R},{10:R},{11:R}",h.step,i,h.loss,h.nrmse,l.position.x,l.position.y,l.position.z,l.intensity,l.cctKelvin,l.direction.x,l.direction.y,l.direction.z));
                }
            }
        }

        private static GradientReport CheckGradient(string name,LightSpec spec,LightSpec reference,SurfaceSampleSet samples)
        {
            LightSpec[] rig={spec};float[] x=AnalyticLighting.ToVector(rig),analytic=new float[x.Length];Vector3[] target=AnalyticLighting.Evaluate(new[]{reference},samples);
            AnalyticLighting.LossAndGradient(x,rig,target,samples,true,true,true,true,analytic);return FiniteDifference(name,rig,x,analytic,target,samples,false,null);
        }
        private static GradientReport CheckMultiGradient(LightSpec[] targetRig,LightSpec[] initial,SurfaceSampleSet samples,DiffuseRadiosity transport=null)
        {
            float[] targetMask=UnityVisibility.Evaluate(targetRig,samples);Vector3[] target=AnalyticLighting.EvaluateField(targetRig,samples,targetMask,true,transport);
            float[] x=AnalyticLighting.ToVector(initial,true),analytic=new float[x.Length];float[] mask=UnityVisibility.Evaluate(initial,samples);
            AnalyticLighting.LossAndGradient(x,initial,target,samples,true,true,false,true,analytic,mask,true,transport);
            return FiniteDifference(transport==null?"three_light_cct_shadow":"three_light_radiosity",initial,x,analytic,target,samples,true,mask,transport);
        }
        private static GradientReport FiniteDifference(string name,LightSpec[] rig,float[] x,float[] analytic,Vector3[] target,SurfaceSampleSet samples,bool cct,float[] fixedMask,DiffuseRadiosity transport=null)
        {
            var finite=new float[x.Length];const float h=1e-4f;bool masksFixed=true;
            for(int i=0;i<x.Length;i++)
            {
                int parameter=i%AnalyticLighting.ParametersPerLight;
                if(parameter>=4&&parameter<=6&&rig[i/AnalyticLighting.ParametersPerLight].kind!=LightKind.Spot)continue;
                if(cct&&parameter>=8)continue;
                // Finite-difference the same detached objective as the analytic derivative.
                float original=x[i];x[i]=original+h;float plus=AnalyticLighting.Loss(x,rig,target,samples,fixedMask,cct,transport);
                if(fixedMask!=null&&parameter<3)CheckMask();
                x[i]=original-h;float minus=AnalyticLighting.Loss(x,rig,target,samples,fixedMask,cct,transport);
                if(fixedMask!=null&&parameter<3)CheckMask();
                x[i]=original;
                finite[i]=(plus-minus)/(2*h);
            }
            double aa=0,ff=0,af=0,diff=0,max=0;
            for(int i=0;i<x.Length;i++){aa+=analytic[i]*analytic[i];ff+=finite[i]*finite[i];af+=analytic[i]*finite[i];diff+=(analytic[i]-finite[i])*(analytic[i]-finite[i]);max=Math.Max(max,Math.Abs(analytic[i]-finite[i]));}
            float cosine=(float)(af/Math.Max(Math.Sqrt(aa*ff),1e-20)),relative=(float)(Math.Sqrt(diff)/Math.Max(Math.Sqrt(aa),Math.Sqrt(ff)));
            return new GradientReport { name=name,maxAbsoluteError=(float)max,relativeNormError=relative,cosine=cosine,allMasksFixed=masksFixed,passed=cosine>0.999f&&relative<0.02f };

            void CheckMask()
            {
                float[] trial=UnityVisibility.Evaluate(AnalyticLighting.FromVector(x,rig,cct),samples);
                for(int j=0;j<trial.Length;j++)if(trial[j]!=fixedMask[j]){masksFixed=false;break;}
            }
        }

        private static RadiosityReport CheckRadiosity(DiffuseRadiosity transport,LightSpec[] rig,SurfaceSampleSet samples)
        {
            Vector3[] irradiance=AnalyticLighting.Evaluate(rig,samples,UnityVisibility.Evaluate(rig,samples),true);
            Vector3[] direct=AnalyticLighting.DiffuseRadiance(irradiance,samples),full=transport.Evaluate(irradiance);
            Vector3[] Bounce(Vector3[] last)
            {
                var next=transport.Reflect(last);
                for(int i=0;i<next.Length;i++)next[i]+=direct[i];
                return next;
            }
            var report=new RadiosityReport
            {
                patches=samples.Count,blockedPairs=transport.BlockedPairs,refinedPairs=transport.RefinedPairs,maxRawRowSum=transport.MaxRawRowSum,
                maxRowSum=transport.MaxRowSum,maxReciprocityError=transport.MaxReciprocityError,
                fixedPointNrmse=AnalyticLighting.FieldNrmse(Bounce(full),full,samples),
                zeroBounceNrmse=AnalyticLighting.FieldNrmse(direct,full,samples)
            };
            Vector3[] estimate=direct;
            for(int i=1;i<=16;i++)
            {
                estimate=Bounce(estimate);
                if(i==1)report.oneBounceNrmse=AnalyticLighting.FieldNrmse(estimate,full,samples);
                if(i==4)report.fourBounceNrmse=AnalyticLighting.FieldNrmse(estimate,full,samples);
            }
            report.sixteenBounceNrmse=AnalyticLighting.FieldNrmse(estimate,full,samples);
            if(!(report.maxRowSum<1&&report.maxReciprocityError<1e-6f&&report.blockedPairs>0&&report.refinedPairs>0&&report.fixedPointNrmse<1e-5f&&
                report.zeroBounceNrmse>0.05f&&report.oneBounceNrmse<report.zeroBounceNrmse&&report.fourBounceNrmse<report.oneBounceNrmse&&
                report.sixteenBounceNrmse<0.01f))throw new Exception("Diffuse radiosity transport failed validation: "+JsonUtility.ToJson(report));
            return report;
        }

        private static void ValidateGpu(string name,LightSpec[] rig,SurfaceSampleSet samples,MvpReport report,bool shadows=false)
        {
            var shader=AssetDatabase.LoadAssetAtPath<ComputeShader>(ComputePath);
            float[] mask=shadows?UnityVisibility.Evaluate(rig,samples):null;
            Vector3[] cpuE=AnalyticLighting.Evaluate(rig,samples,mask,shadows),cpuR=AnalyticLighting.DiffuseRadiance(cpuE,samples);
            if(shadows)
            {
                report.blockedValidationRays=CountBlocked(mask);
                report.shadowFieldNrmse=AnalyticLighting.FieldNrmse(AnalyticLighting.Evaluate(rig,samples,null,true),cpuE,samples);
                if(report.blockedValidationRays==0||report.shadowFieldNrmse<0.01f)throw new InvalidOperationException("Cornell scene has no effective collider shadows.");
            }
            using(var gpu=new SurfaceLightingGpu(shader))
            {
                gpu.Evaluate(rig,samples,out Vector3[] e,out Vector3[] r,mask,shadows);
                for(int j=0;j<e.Length;j++)
                {
                    float errorE=(e[j]-cpuE[j]).magnitude,errorR=(r[j]-cpuR[j]).magnitude;
                    if(!float.IsFinite(errorE)||!float.IsFinite(errorR))throw new Exception("Non-finite CPU/GPU result for "+name);
                    report.gpuMaxAbsoluteError=Mathf.Max(report.gpuMaxAbsoluteError,errorE);
                    report.gpuMaxRelativeError=Mathf.Max(report.gpuMaxRelativeError,errorE/Mathf.Max(cpuE[j].magnitude,1e-5f));
                    report.gpuRadianceMaxAbsoluteError=Mathf.Max(report.gpuRadianceMaxAbsoluteError,errorR);
                }
                WriteGpuField(name,e,r,samples);
            }
            if(report.gpuMaxRelativeError>1e-4f||report.gpuRadianceMaxAbsoluteError>1e-7f)throw new Exception("CPU/GPU field mismatch for "+name);
        }
        private static void WriteGpuField(string name,Vector3[] e,Vector3[] r,SurfaceSampleSet samples)
        {
            using(var writer=new StreamWriter(Path.Combine(OutputRoot,"gpu_surface_field_"+name+".csv")))
            {
                writer.WriteLine("surface,x,y,z,nx,ny,nz,area,irradiance_r,irradiance_g,irradiance_b,radiance_r,radiance_g,radiance_b");
                for(int j=0;j<e.Length;j++)writer.WriteLine(string.Format(CultureInfo.InvariantCulture,"{0},{1:R},{2:R},{3:R},{4:R},{5:R},{6:R},{7:R},{8:R},{9:R},{10:R},{11:R},{12:R},{13:R}",samples.names[j],samples.positions[j].x,samples.positions[j].y,samples.positions[j].z,samples.normals[j].x,samples.normals[j].y,samples.normals[j].z,samples.areas[j],e[j].x,e[j].y,e[j].z,r[j].x,r[j].y,r[j].z));
            }
        }
        private static void SaveField(string path,SurfaceSampleSet s,Vector3[] target,Vector3[] initial,Vector3[] optimized)
        {
            using(var writer=new StreamWriter(path))
            {
                writer.WriteLine("surface,x,y,z,nx,ny,nz,area,target_r,target_g,target_b,initial_r,initial_g,initial_b,optimized_r,optimized_g,optimized_b");
                for(int j=0;j<s.Count;j++)writer.WriteLine(string.Format(CultureInfo.InvariantCulture,"{0},{1:R},{2:R},{3:R},{4:R},{5:R},{6:R},{7:R},{8:R},{9:R},{10:R},{11:R},{12:R},{13:R},{14:R},{15:R},{16:R}",s.names[j],s.positions[j].x,s.positions[j].y,s.positions[j].z,s.normals[j].x,s.normals[j].y,s.normals[j].z,s.areas[j],target[j].x,target[j].y,target[j].z,initial[j].x,initial[j].y,initial[j].z,optimized[j].x,optimized[j].y,optimized[j].z));
            }
        }

        private static LightSpec PointTarget(){return new LightSpec { position=new Vector3(0.25f,1.35f,0.10f),intensity=2.5f,color=new Color(1,0.85f,0.70f,1) };}
        private static LightSpec PointInitial(){return new LightSpec { position=new Vector3(-0.30f,0.95f,-0.25f),intensity=1.25f,color=new Color(1,0.85f,0.70f,1) };}
        private static LightSpec SpotTarget(){return new LightSpec { position=new Vector3(-0.45f,1.65f,-0.45f),intensity=4,color=new Color(1,0.9f,0.8f,1),kind=LightKind.Spot,direction=new Vector3(0.2f,-1,0.5f).normalized,range=8,innerAngle=25,outerAngle=50 };}
        private static LightSpec SpotInitial(){return new LightSpec { position=new Vector3(0.10f,1.20f,-0.10f),intensity=4,color=new Color(1,0.9f,0.8f,1),kind=LightKind.Spot,direction=new Vector3(0.65f,-0.55f,-0.30f).normalized,range=8,innerAngle=25,outerAngle=50 };}
        private static LightSpec[] MultiTarget()
        {
            var specs=new[]{new LightSpec { position=new Vector3(-0.45f,1.35f,-0.25f),intensity=1.5f,cctKelvin=2700 },new LightSpec { position=new Vector3(0.15f,1.65f,0.15f),intensity=1.2f,cctKelvin=5000 },new LightSpec { position=new Vector3(0.55f,0.85f,0.35f),intensity=1,cctKelvin=8000 }};
            foreach(var s in specs)s.color=AnalyticLighting.CctToRgb(s.cctKelvin);return specs;
        }
        private static LightSpec[] MultiInitial(bool perturbCct)
        {
            var specs=MultiTarget();
            Vector3[] deltas={new Vector3(0.30f,0.21f,-0.24f),new Vector3(0.22f,-0.23f,0.20f),new Vector3(0.17f,0.26f,-0.24f)};
            float[] intensity={0.55f,1.55f,0.65f},cct={4200,3300,5400};
            for(int i=0;i<specs.Length;i++)
            {
                specs[i].position+=deltas[i];specs[i].intensity*=intensity[i];
                if(perturbCct)specs[i].cctKelvin=cct[i];
                specs[i].color=AnalyticLighting.CctToRgb(specs[i].cctKelvin);
            }
            return specs;
        }

        internal static void Capture(string filename)
        {
            var camera=GameObject.Find("Validation Camera").GetComponent<Camera>();
            var rt=RenderTexture.GetTemporary(768,768,24,RenderTextureFormat.ARGB32,RenderTextureReadWrite.sRGB);
            var previous=RenderTexture.active;var previousTarget=camera.targetTexture;
            var texture=new Texture2D(768,768,TextureFormat.RGB24,false);
            try
            {
                camera.targetTexture=rt;camera.Render();RenderTexture.active=rt;
                texture.ReadPixels(new Rect(0,0,768,768),0,0);texture.Apply();
                File.WriteAllBytes(Path.Combine(OutputRoot,filename),texture.EncodeToPNG());
            }
            finally
            {
                camera.targetTexture=previousTarget;RenderTexture.active=previous;
                UnityEngine.Object.DestroyImmediate(texture);RenderTexture.ReleaseTemporary(rt);
            }
        }
    }
}
