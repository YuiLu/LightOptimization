using UnityEngine;

namespace LightingMvp
{
    /// <summary>Scene-adapter boundary between optimizer parameters and editable Unity lights.</summary>
    public static class UnityLightAdapter
    {
        public static void Apply(Light light,LightSpec spec,bool shadows=false)
        {
            light.transform.position=spec.position;
            light.type=spec.kind==LightKind.Point?LightType.Point:LightType.Spot;
            light.transform.rotation=spec.kind==LightKind.Spot?
                Quaternion.FromToRotation(Vector3.forward,spec.direction.normalized):Quaternion.identity;
            light.color=spec.color;light.intensity=spec.intensity;light.range=spec.range;light.shadows=shadows?LightShadows.Hard:LightShadows.None;
            light.colorTemperature=spec.cctKelvin;light.useColorTemperature=false;light.bounceIntensity=0;
            if(spec.kind==LightKind.Spot){light.spotAngle=spec.outerAngle*2;light.innerSpotAngle=spec.innerAngle*2;}
        }

        public static LightSpec FromUnity(Light light)
        {
            return new LightSpec
            {
                position=light.transform.position,intensity=light.intensity,color=light.color,
                kind=light.type==LightType.Spot?LightKind.Spot:LightKind.Point,direction=light.transform.forward,
                range=light.range,innerAngle=light.innerSpotAngle*0.5f,outerAngle=light.spotAngle*0.5f,cctKelvin=light.colorTemperature,
            };
        }
    }
}
