using System;
using UnityEngine;

namespace LightingMvp
{
    /// <summary>Detached hard visibility, laid out as [light * sampleCount + sample].</summary>
    public delegate float[] VisibilityProvider(LightSpec[] lights,SurfaceSampleSet samples);

    /// <summary>Multi-light direct irradiance, CCT chromaticity, and analytic derivatives.</summary>
    public static class AnalyticLighting
    {
        public const int ParametersPerLight=10; // position(3), logIntensity(1), rawDirection(3), color/CCT(3)

        public static Vector3[] Evaluate(LightSpec[] lights,SurfaceSampleSet samples,float[] visibility=null,bool useCct=false)
        {
            var result=new Vector3[samples.Count];
            var colors=new Vector3[lights.Length];
            for(int l=0;l<lights.Length;l++)colors[l]=Chromaticity(lights[l],useCct);
            for(int j=0;j<samples.Count;j++)for(int l=0;l<lights.Length;l++)
            {
                float v=visibility==null?1:visibility[l*samples.Count+j];
                if(v!=0)result[j]+=v*TermsFor(lights[l],samples.positions[j],samples.normals[j]).strength*colors[l];
            }
            return result;
        }

        public static Vector3[] DiffuseRadiance(Vector3[] irradiance,SurfaceSampleSet samples)
        {
            var result=new Vector3[irradiance.Length];
            for(int j=0;j<result.Length;j++)result[j]=Vector3.Scale(irradiance[j],RGB(samples.albedos[j]))/Mathf.PI;
            return result;
        }

        public static Vector3[] EvaluateField(LightSpec[] lights,SurfaceSampleSet samples,float[] visibility,bool useCct,DiffuseRadiosity transport)
        {
            Vector3[] direct=Evaluate(lights,samples,visibility,useCct);
            return transport==null?direct:transport.Evaluate(direct);
        }

        public static float WeightedLoss(Vector3[] prediction,Vector3[] target,SurfaceSampleSet samples)
        {
            float sum=0;for(int j=0;j<prediction.Length;j++)sum+=samples.areas[j]*(prediction[j]-target[j]).sqrMagnitude;
            return sum/(TotalArea(samples)*3);
        }

        public static float FieldNrmse(Vector3[] prediction,Vector3[] target,SurfaceSampleSet samples)
        {
            float numerator=0,denominator=0;
            for(int j=0;j<prediction.Length;j++){numerator+=samples.areas[j]*(prediction[j]-target[j]).sqrMagnitude;denominator+=samples.areas[j]*target[j].sqrMagnitude;}
            return Mathf.Sqrt(numerator/Mathf.Max(denominator,1e-20f));
        }

        public static float Loss(float[] vector,LightSpec[] template,Vector3[] target,SurfaceSampleSet samples,float[] visibility=null,bool useCct=false,DiffuseRadiosity transport=null)
        {
            return WeightedLoss(EvaluateField(FromVector(vector,template,useCct),samples,visibility,useCct,transport),target,samples);
        }

        public static float LossAndGradient(float[] vector,LightSpec[] template,Vector3[] target,SurfaceSampleSet samples,
            bool position,bool intensity,bool direction,bool color,float[] gradient,float[] visibility=null,bool useCct=false,DiffuseRadiosity transport=null)
        {
            Array.Clear(gradient,0,gradient.Length);LightSpec[] lights=FromVector(vector,template,useCct);
            float totalArea=TotalArea(samples),loss=0;
            var colors=new Vector3[lights.Length];
            var derivatives=color&&useCct?new Vector3[lights.Length]:null;
            for(int l=0;l<lights.Length;l++)
            {
                colors[l]=Chromaticity(lights[l],useCct);
                if(derivatives!=null)derivatives[l]=CctDerivative(lights[l].cctKelvin);
            }
            Vector3[] prediction=EvaluateField(lights,samples,visibility,useCct,transport);
            var fieldGradient=new Vector3[samples.Count];
            for(int j=0;j<samples.Count;j++)
            {
                Vector3 error=prediction[j]-target[j];loss+=samples.areas[j]*error.sqrMagnitude/(totalArea*3);
                fieldGradient[j]=(2*samples.areas[j]/(totalArea*3))*error;
            }
            // Only the fixed diffuse transport is differentiated here; the per-light hard mask stays detached.
            Vector3[] directGradient=transport==null?fieldGradient:transport.Backpropagate(fieldGradient);
            for(int j=0;j<samples.Count;j++)
            {
                Vector3 upstream=directGradient[j];
                for(int l=0;l<lights.Length;l++)
                {
                    float v=visibility==null?1:visibility[l*samples.Count+j];if(v==0)continue;
                    LightSpec light=lights[l];Terms t=TermsFor(light,samples.positions[j],samples.normals[j]);t.strength*=v;
                    float scalar=Vector3.Dot(upstream,colors[l]);int offset=l*ParametersPerLight;
                    if(position)Add(gradient,offset,v*scalar*PositionDerivative(light,t,samples.normals[j]));
                    if(intensity)gradient[offset+3]+=scalar*t.strength;
                    if(direction&&light.kind==LightKind.Spot)Add(gradient,offset+4,v*scalar*RawDirectionDerivative(light,t));
                    if(color)
                    {
                        if(useCct)gradient[offset+7]+=t.strength*Vector3.Dot(upstream,derivatives[l]);
                        else
                        {
                            gradient[offset+7]+=upstream.x*t.strength;
                            gradient[offset+8]+=upstream.y*t.strength;
                            gradient[offset+9]+=upstream.z*t.strength;
                        }
                    }
                }
            }
            return loss;
        }

        public static float[] ToVector(LightSpec[] lights,bool useCct=false)
        {
            var v=new float[lights.Length*ParametersPerLight];
            for(int l=0;l<lights.Length;l++)
            {
                int i=l*ParametersPerLight;LightSpec light=lights[l];
                v[i]=light.position.x;v[i+1]=light.position.y;v[i+2]=light.position.z;v[i+3]=Mathf.Log(Mathf.Max(light.intensity,1e-8f));
                v[i+4]=light.direction.x;v[i+5]=light.direction.y;v[i+6]=light.direction.z;
                if(useCct){v[i+7]=Mathf.Log(Mathf.Clamp(light.cctKelvin,1000,40000));}
                else{v[i+7]=light.color.r;v[i+8]=light.color.g;v[i+9]=light.color.b;}
            }
            return v;
        }

        public static LightSpec[] FromVector(float[] v,LightSpec[] template,bool useCct=false)
        {
            LightSpec[] lights=LightSpec.CloneArray(template);
            for(int l=0;l<lights.Length;l++)
            {
                int i=l*ParametersPerLight;lights[l].position=new Vector3(v[i],v[i+1],v[i+2]);
                lights[l].intensity=Mathf.Exp(Mathf.Clamp(v[i+3],-20,20));lights[l].direction=new Vector3(v[i+4],v[i+5],v[i+6]);
                if(useCct){lights[l].cctKelvin=Mathf.Exp(Mathf.Clamp(v[i+7],Mathf.Log(1000),Mathf.Log(40000)));lights[l].color=CctToRgb(lights[l].cctKelvin);}
                else lights[l].color=new Color(v[i+7],v[i+8],v[i+9],1);
            }
            return lights;
        }

        public static Vector3 Chromaticity(LightSpec light,bool useCct) { return useCct?RGB(CctToRgb(light.cctKelvin)):RGB(light.color); }

        public static Color CctToRgb(float kelvin)
        {
            float t=Mathf.Clamp(kelvin/100,10,400),r,g,b;
            if(t<=66){r=1;g=0.3900816f*Mathf.Log(t)-0.6318414f;}else{r=1.292936f*Mathf.Pow(t-60,-0.1332047f);g=1.1298909f*Mathf.Pow(t-60,-0.0755147f);}
            if(t>=66)b=1;else if(t<=19)b=0;else b=0.5432068f*Mathf.Log(t-10)-1.196254f;
            r=Mathf.Clamp(r,1e-6f,1);g=Mathf.Clamp(g,1e-6f,1);b=Mathf.Clamp(b,1e-6f,1);float sum=Mathf.Max(r+g+b,1e-12f);
            return new Color(r/sum,g/sum,b/sum,1);
        }

        private static Vector3 CctDerivative(float kelvin)
        {
            float t=Mathf.Clamp(kelvin/100,10,400),r,g,b,dr,dg,db;
            if(t<=66){r=1;g=0.3900816f*Mathf.Log(t)-0.6318414f;dr=0;dg=0.3900816f;}
            else{r=1.292936f*Mathf.Pow(t-60,-0.1332047f);g=1.1298909f*Mathf.Pow(t-60,-0.0755147f);dr=-0.1332047f*r*t/(t-60);dg=-0.0755147f*g*t/(t-60);}
            if(t>=66){b=1;db=0;}else if(t<=19){b=0;db=0;}else{b=0.5432068f*Mathf.Log(t-10)-1.196254f;db=0.5432068f*t/(t-10);}
            r=Mathf.Clamp(r,1e-6f,1);g=Mathf.Clamp(g,1e-6f,1);b=Mathf.Clamp(b,1e-6f,1);
            if(r<=1e-6f||r>=1)dr=0;if(g<=1e-6f||g>=1)dg=0;if(b<=1e-6f||b>=1)db=0;
            float sum=r+g+b,ds=dr+dg+db;
            return new Vector3((dr*sum-r*ds)/(sum*sum),(dg*sum-g*ds)/(sum*sum),(db*sum-b*ds)/(sum*sum));
        }

        private struct Terms { public Vector3 delta,omega;public float distance2,baseAttenuation,cosine,spot,coneT,coneCosine,strength; }
        private static Terms TermsFor(LightSpec light,Vector3 x,Vector3 n)
        {
            Terms t=new Terms();t.delta=light.position-x;t.distance2=Mathf.Max(t.delta.sqrMagnitude,1e-12f);float distance=Mathf.Sqrt(t.distance2);t.omega=t.delta/distance;
            t.cosine=Mathf.Max(0,Vector3.Dot(n,t.omega));float window=Mathf.Max(0,1-t.distance2/(light.range*light.range));
            t.baseAttenuation=light.attenuation/(light.attenuation+t.distance2)*window*window;t.spot=1;t.coneT=1;
            if(light.kind==LightKind.Spot)
            {
                t.coneCosine=Vector3.Dot(-t.omega,light.direction.normalized);t.coneT=(t.coneCosine-CosOuter(light))/Mathf.Max(CosInner(light)-CosOuter(light),1e-8f);
                t.spot=Mathf.Clamp01(t.coneT)*Mathf.Clamp01(t.coneT);
            }
            t.strength=light.intensity*t.baseAttenuation*t.cosine*t.spot;return t;
        }
        private static Vector3 PositionDerivative(LightSpec light,Terms t,Vector3 n)
        {
            float distance=Mathf.Sqrt(t.distance2);Vector3 dBase=Vector3.zero,dCos=Vector3.zero,dSpot=Vector3.zero;
            float window=Mathf.Max(0,1-t.distance2/(light.range*light.range));
            if(t.baseAttenuation>0&&window>0)dBase=t.baseAttenuation*(-2*t.delta/(light.attenuation+t.distance2)-4*t.delta/(light.range*light.range*window));
            if(t.cosine>0)dCos=n/distance-t.delta*t.cosine/t.distance2;
            if(light.kind==LightKind.Spot&&t.coneT>0&&t.coneT<1)
            {
                Vector3 d=light.direction.normalized,dCone=-d/distance-t.delta*t.coneCosine/t.distance2;
                dSpot=2*t.coneT*dCone/Mathf.Max(CosInner(light)-CosOuter(light),1e-8f);
            }
            return light.intensity*(dBase*t.cosine*t.spot+t.baseAttenuation*dCos*t.spot+t.baseAttenuation*t.cosine*dSpot);
        }
        private static Vector3 RawDirectionDerivative(LightSpec light,Terms t)
        {
            if(t.coneT<=0||t.coneT>=1)return Vector3.zero;float denominator=Mathf.Max(CosInner(light)-CosOuter(light),1e-8f);
            Vector3 raw=light.direction;float length=Mathf.Max(raw.magnitude,1e-12f);Vector3 d=raw/length;
            Vector3 dNormalized=2*t.coneT*(-t.omega)/denominator,dRaw=(dNormalized-d*Vector3.Dot(d,dNormalized))/length;
            return light.intensity*t.baseAttenuation*t.cosine*dRaw;
        }
        private static float CosInner(LightSpec l){return Mathf.Cos(l.innerAngle*Mathf.Deg2Rad);}
        private static float CosOuter(LightSpec l){return Mathf.Cos(l.outerAngle*Mathf.Deg2Rad);}
        private static Vector3 RGB(Color c){return new Vector3(c.r,c.g,c.b);}
        private static void Add(float[] a,int i,Vector3 v){a[i]+=v.x;a[i+1]+=v.y;a[i+2]+=v.z;}
        private static float TotalArea(SurfaceSampleSet s){float a=0;for(int i=0;i<s.areas.Length;i++)a+=s.areas[i];return a;}
    }
}
