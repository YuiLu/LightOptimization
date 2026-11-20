Shader "LightingMvp/IrradianceOverlay"
{
    Properties { _Alpha("Overlay Opacity",Range(0,1)) = 1 }
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" "RenderPipeline"="UniversalPipeline" }
        Pass
        {
            Tags { "LightMode"="UniversalForward" }
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off
            ZTest LEqual
            Offset -1,-1
            Cull Off
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            float _Alpha;
            struct Attributes { float3 positionWS : POSITION; float4 color : COLOR; };
            struct Varyings { float4 positionCS : SV_POSITION; float4 color : COLOR; };
            Varyings Vert(Attributes input)
            {
                Varyings output;
                output.positionCS=TransformWorldToHClip(input.positionWS);
                output.color=input.color;
                return output;
            }
            half4 Frag(Varyings input) : SV_Target { return half4(input.color.rgb,input.color.a*_Alpha); }
            ENDHLSL
        }
    }
}
