using System;
using System.Runtime.InteropServices;
using UnityEngine;

namespace LightingMvp
{
    /// <summary>GPU multi-light direct irradiance and Lambertian outgoing radiance backend.</summary>
    public sealed class SurfaceLightingGpu : IDisposable
    {
        [StructLayout(LayoutKind.Sequential)] private struct SampleGpu { public Vector3 position;public float area;public Vector3 normal;public float padding;public Color albedo; }
        [StructLayout(LayoutKind.Sequential)] private struct LightGpu
        { public Vector3 position;public float intensity;public Vector3 direction;public float range;public Color color;public float attenuation,cosInner,cosOuter;public int kind; }
        private readonly ComputeShader shader;
        public SurfaceLightingGpu(ComputeShader shader)
        {
            if(shader==null)throw new ArgumentNullException("shader");if(!SystemInfo.supportsComputeShaders)throw new NotSupportedException("Compute shaders are unavailable");this.shader=shader;
        }
        public void Dispose() { }

        public void Evaluate(LightSpec[] lights,SurfaceSampleSet samples,out Vector3[] irradiance,out Vector3[] radiance,float[] visibility=null,bool useCct=false)
        {
            if(visibility!=null&&visibility.Length!=lights.Length*samples.Count)throw new ArgumentException("Visibility must be [lights*samples]");
            if(lights.Length==0||samples.Count==0)
            {
                irradiance=new Vector3[samples.Count];radiance=new Vector3[samples.Count];return;
            }
            var sg=new SampleGpu[samples.Count];var lg=new LightGpu[lights.Length];
            for(int j=0;j<sg.Length;j++)sg[j]=new SampleGpu { position=samples.positions[j],area=samples.areas[j],normal=samples.normals[j],albedo=samples.albedos[j] };
            for(int l=0;l<lg.Length;l++)
            {
                LightSpec s=lights[l];Color color=useCct?AnalyticLighting.CctToRgb(s.cctKelvin):s.color;
                lg[l]=new LightGpu { position=s.position,intensity=s.intensity,direction=s.direction.normalized,range=s.range,color=color,attenuation=s.attenuation,
                    cosInner=Mathf.Cos(s.innerAngle*Mathf.Deg2Rad),cosOuter=Mathf.Cos(s.outerAngle*Mathf.Deg2Rad),kind=(int)s.kind };
            }
            ComputeBuffer sb=null,lb=null,vb=null,eb=null,rb=null;
            try
            {
                sb=new ComputeBuffer(sg.Length,Marshal.SizeOf(typeof(SampleGpu)));sb.SetData(sg);lb=new ComputeBuffer(lg.Length,Marshal.SizeOf(typeof(LightGpu)));lb.SetData(lg);
                eb=new ComputeBuffer(sg.Length,sizeof(float)*4);rb=new ComputeBuffer(sg.Length,sizeof(float)*4);
                int kernel=shader.FindKernel("EvaluateSurfaceLighting");shader.SetInt("_SampleCount",sg.Length);shader.SetInt("_LightCount",lg.Length);shader.SetInt("_UseVisibility",visibility!=null?1:0);
                shader.SetBuffer(kernel,"_Samples",sb);shader.SetBuffer(kernel,"_Lights",lb);shader.SetBuffer(kernel,"_Irradiance",eb);shader.SetBuffer(kernel,"_OutgoingRadiance",rb);
                // DX11 requires all declared buffers to be bound, even when the branch is disabled.
                vb=new ComputeBuffer(visibility?.Length??1,sizeof(float));vb.SetData(visibility??new[]{1f});shader.SetBuffer(kernel,"_Visibility",vb);
                shader.GetKernelThreadGroupSizes(kernel,out uint x,out uint _,out uint _);shader.Dispatch(kernel,Mathf.CeilToInt(sg.Length/(float)x),1,1);
                var e=new Vector4[sg.Length];var r=new Vector4[sg.Length];eb.GetData(e);rb.GetData(r);irradiance=new Vector3[e.Length];radiance=new Vector3[e.Length];
                for(int j=0;j<e.Length;j++){irradiance[j]=new Vector3(e[j].x,e[j].y,e[j].z);radiance[j]=new Vector3(r[j].x,r[j].y,r[j].z);}
            }
            finally{if(sb!=null)sb.Dispose();if(lb!=null)lb.Dispose();if(vb!=null)vb.Dispose();if(eb!=null)eb.Dispose();if(rb!=null)rb.Dispose();}
        }
    }
}
