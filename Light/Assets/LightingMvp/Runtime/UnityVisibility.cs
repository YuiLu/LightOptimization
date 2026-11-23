using UnityEngine;

namespace LightingMvp
{
    /// <summary>Unity collider-backed hard visibility; the mask is detached from gradients.</summary>
    public static class UnityVisibility
    {
        public static float[] Evaluate(LightSpec[] lights,SurfaceSampleSet samples)
        {
            return Evaluate(lights,samples,1e-4f,~0);
        }

        public static float[] Evaluate(LightSpec[] lights,SurfaceSampleSet samples,float epsilon,int layerMask)
        {
            Physics.SyncTransforms();
            var result=new float[lights.Length*samples.Count];
            for(int l=0;l<lights.Length;l++)for(int j=0;j<samples.Count;j++)
            {
                Vector3 origin=samples.positions[j]+samples.normals[j]*epsilon;
                Vector3 delta=lights[l].position-origin;float distance=delta.magnitude;
                bool visible=distance>2*epsilon&&float.IsFinite(distance);
                // Start at the light so floor receivers under solids are shadowed. Unity
                // would omit the hit if a receiver started inside the block.
                if(visible)visible=!Physics.Raycast(lights[l].position,-delta/distance,distance-epsilon,layerMask,QueryTriggerInteraction.Ignore);
                result[l*samples.Count+j]=visible?1:0;
            }
            return result;
        }

        public static float Mismatch(LightSpec[] a,LightSpec[] b,SurfaceSampleSet samples)
        {
            float[] va=Evaluate(a,samples),vb=Evaluate(b,samples);int different=0;
            for(int i=0;i<va.Length;i++)if(va[i]!=vb[i])different++;
            return different/(float)va.Length;
        }
    }
}
