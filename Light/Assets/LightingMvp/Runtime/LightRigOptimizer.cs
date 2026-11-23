using System;
using System.Collections.Generic;
using UnityEngine;

namespace LightingMvp
{
    [Serializable] public sealed class OptimizerOptions
    {
        public int steps=500,lbfgsIterations=200;
        public bool position=true,intensity=true,direction,color,cct;
        public float positionRate=0.035f,intensityRate=0.045f,directionRate=0.04f,colorRate=0.03f;
    }

    [Serializable] public sealed class OptimizationHistoryEntry
    {
        public int step;public float loss,nrmse;public LightSpec[] lights;
        public OptimizationHistoryEntry() { }
        public OptimizationHistoryEntry(int step,float loss,float nrmse,LightSpec[] lights){this.step=step;this.loss=loss;this.nrmse=nrmse;this.lights=lights;}
    }

    public sealed class OptimizationResult
    {
        public LightSpec[] initial,optimized;
        public float initialLoss,finalLoss,initialNrmse,finalNrmse;
        public int lbfgsEvaluations;
        public readonly List<OptimizationHistoryEntry> history=new List<OptimizationHistoryEntry>();
    }

    /// <summary>Renderer-independent multi-light Adam + L-BFGS with detached visibility.</summary>
    public static class LightRigOptimizer
    {
        public static OptimizationResult Optimize(LightSpec[] initial,Vector3[] target,SurfaceSampleSet samples,OptimizerOptions options,VisibilityProvider visibility=null,DiffuseRadiosity transport=null)
        {
            float[] x=AnalyticLighting.ToVector(initial,options.cct),m=new float[x.Length],v=new float[x.Length],g=new float[x.Length];
            Vector3[] initialPrediction=EvaluateCurrent(initial,samples,options,visibility,transport);
            var result=new OptimizationResult { initial=LightSpec.CloneArray(initial),initialLoss=AnalyticLighting.WeightedLoss(initialPrediction,target,samples),initialNrmse=AnalyticLighting.FieldNrmse(initialPrediction,target,samples) };
            Record(result,x,initial,target,samples,0,options,visibility,transport);
            for(int step=1;step<=options.steps;step++)
            {
                LightSpec[] current=AnalyticLighting.FromVector(x,initial,options.cct);float[] mask=visibility?.Invoke(current,samples);
                AnalyticLighting.LossAndGradient(x,initial,target,samples,options.position,options.intensity,options.direction,options.color,g,mask,options.cct,transport);
                for(int i=0;i<x.Length;i++)
                {
                    float rate=RateForIndex(i%AnalyticLighting.ParametersPerLight,options);if(rate==0)continue;
                    m[i]=0.9f*m[i]+0.1f*g[i];v[i]=0.999f*v[i]+0.001f*g[i]*g[i];
                    float mHat=m[i]/(1-Mathf.Pow(0.9f,step)),vHat=v[i]/(1-Mathf.Pow(0.999f,step));x[i]-=rate*mHat/(Mathf.Sqrt(vHat)+1e-8f);
                }
                Record(result,x,initial,target,samples,step,options,visibility,transport);
            }
            if(options.lbfgsIterations>0&&AnalyticLighting.FieldNrmse(EvaluateCurrent(AnalyticLighting.FromVector(x,initial,options.cct),samples,options,visibility,transport),target,samples)>1e-5f)
                RefineLbfgs(x,initial,target,samples,options,result,visibility,transport);
            result.optimized=AnalyticLighting.FromVector(x,initial,options.cct);Vector3[] prediction=EvaluateCurrent(result.optimized,samples,options,visibility,transport);
            result.finalLoss=AnalyticLighting.WeightedLoss(prediction,target,samples);result.finalNrmse=AnalyticLighting.FieldNrmse(prediction,target,samples);
            return result;
        }

        private static Vector3[] EvaluateCurrent(LightSpec[] lights,SurfaceSampleSet samples,OptimizerOptions options,VisibilityProvider visibility,DiffuseRadiosity transport)
        {
            return AnalyticLighting.EvaluateField(lights,samples,visibility?.Invoke(lights,samples),options.cct,transport);
        }

        private static void RefineLbfgs(float[] x,LightSpec[] template,Vector3[] target,SurfaceSampleSet samples,OptimizerOptions options,OptimizationResult result,VisibilityProvider visibility,DiffuseRadiosity transport)
        {
            const int memory=10;var sHistory=new List<float[]>();var yHistory=new List<float[]>();
            float objectiveScale=1/Mathf.Max(AnalyticLighting.WeightedLoss(target,new Vector3[target.Length],samples),1e-20f);
            float[] g=new float[x.Length];LightSpec[] current=AnalyticLighting.FromVector(x,template,options.cct);float[] mask=visibility?.Invoke(current,samples);
            float loss=AnalyticLighting.LossAndGradient(x,template,target,samples,options.position,options.intensity,options.direction,options.color,g,mask,options.cct,transport);
            loss*=objectiveScale;Scale(g,objectiveScale);
            for(int iteration=0;iteration<options.lbfgsIterations;iteration++)
            {
                if(Norm(g)<1e-12f)break;float[] q=(float[])g.Clone(),alpha=new float[sHistory.Count];
                for(int k=sHistory.Count-1;k>=0;k--){alpha[k]=Dot(sHistory[k],q)/Dot(yHistory[k],sHistory[k]);AddScaled(q,yHistory[k],-alpha[k]);}
                float gamma=sHistory.Count>0?Dot(yHistory[sHistory.Count-1],sHistory[sHistory.Count-1])/Mathf.Max(Dot(yHistory[sHistory.Count-1],yHistory[sHistory.Count-1]),1e-30f):1;
                float[] r=new float[x.Length];for(int i=0;i<x.Length;i++)r[i]=gamma*q[i];
                for(int k=0;k<sHistory.Count;k++){float beta=Dot(yHistory[k],r)/Dot(yHistory[k],sHistory[k]);AddScaled(r,sHistory[k],alpha[k]-beta);}
                float[] direction=new float[x.Length];float directionalDerivative=0;
                for(int i=0;i<x.Length;i++){if(RateForIndex(i%AnalyticLighting.ParametersPerLight,options)==0)continue;direction[i]=-r[i];directionalDerivative+=g[i]*direction[i];}
                if(directionalDerivative>=0)
                {
                    sHistory.Clear();yHistory.Clear();directionalDerivative=0;
                    for(int i=0;i<x.Length;i++)if(RateForIndex(i%AnalyticLighting.ParametersPerLight,options)!=0){direction[i]=-g[i];directionalDerivative-=g[i]*g[i];}
                }
                float step=1;bool accepted=false;float[] candidate=new float[x.Length],candidateGradient=new float[x.Length];float candidateLoss=0;
                for(int line=0;line<35;line++)
                {
                    Array.Copy(x,candidate,x.Length);for(int i=0;i<x.Length;i++)candidate[i]+=step*direction[i];
                    current=AnalyticLighting.FromVector(candidate,template,options.cct);mask=visibility?.Invoke(current,samples);
                    candidateLoss=AnalyticLighting.LossAndGradient(candidate,template,target,samples,options.position,options.intensity,options.direction,options.color,candidateGradient,mask,options.cct,transport);
                    candidateLoss*=objectiveScale;Scale(candidateGradient,objectiveScale);result.lbfgsEvaluations++;
                    if(float.IsFinite(candidateLoss)&&candidateLoss<=loss+1e-4f*step*directionalDerivative){accepted=true;break;}step*=0.5f;
                }
                if(!accepted)break;float[] s=new float[x.Length],y=new float[x.Length];
                for(int i=0;i<x.Length;i++){s[i]=candidate[i]-x[i];y[i]=candidateGradient[i]-g[i];}
                if(Dot(y,s)>1e-20f){sHistory.Add(s);yHistory.Add(y);if(sHistory.Count>memory){sHistory.RemoveAt(0);yHistory.RemoveAt(0);}}
                Array.Copy(candidate,x,x.Length);Array.Copy(candidateGradient,g,g.Length);loss=candidateLoss;
                Record(result,x,template,target,samples,options.steps+result.lbfgsEvaluations,options,visibility,transport);
                if(loss<1e-20f)break;
            }
        }

        private static void Record(OptimizationResult result,float[] x,LightSpec[] template,Vector3[] target,SurfaceSampleSet samples,int step,OptimizerOptions options,VisibilityProvider visibility,DiffuseRadiosity transport)
        {
            LightSpec[] lights=AnalyticLighting.FromVector(x,template,options.cct);Vector3[] p=EvaluateCurrent(lights,samples,options,visibility,transport);
            result.history.Add(new OptimizationHistoryEntry(step,AnalyticLighting.WeightedLoss(p,target,samples),AnalyticLighting.FieldNrmse(p,target,samples),lights));
        }
        private static float Dot(float[] a,float[] b){double sum=0;for(int i=0;i<a.Length;i++)sum+=(double)a[i]*b[i];return(float)sum;}
        private static float Norm(float[] a){return Mathf.Sqrt(Dot(a,a));}
        private static void Scale(float[] a,float scale){for(int i=0;i<a.Length;i++)a[i]*=scale;}
        private static void AddScaled(float[] target,float[] source,float scale){for(int i=0;i<target.Length;i++)target[i]+=scale*source[i];}
        private static float RateForIndex(int i,OptimizerOptions o)
        {
            if(i<3)return o.position?o.positionRate:0;if(i==3)return o.intensity?o.intensityRate:0;
            if(i<7)return o.direction?o.directionRate:0;return o.color?o.colorRate:0;
        }
    }
}
