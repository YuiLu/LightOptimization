using System;
using System.Collections.Generic;
using UnityEngine;

namespace LightingMvp
{
    /// <summary>Camera-independent surface samples read from scene assets, never procedural scene geometry.</summary>
    public sealed class SurfaceSampleSet
    {
        public Vector3[] positions,normals;
        public float[] areas;
        public Color[] albedos;
        public string[] names;
        public int Count { get { return positions.Length; } }

        /// <summary>Samples the quad surfaces of the saved experiment scene in world space.</summary>
        public static SurfaceSampleSet FromRenderers(IEnumerable<MeshRenderer> renderers,int resolution)
        {
            if(resolution<1)throw new ArgumentOutOfRangeException("resolution");
            var p=new List<Vector3>();var n=new List<Vector3>();var a=new List<float>();
            var c=new List<Color>();var names=new List<string>();
            foreach(var renderer in renderers)
            {
                var filter=renderer.GetComponent<MeshFilter>();
                Mesh mesh=filter!=null?filter.sharedMesh:null;
                if(mesh==null)continue;
                if(mesh.vertexCount!=4||mesh.triangles.Length!=6)
                    throw new NotSupportedException("The current scene adapter expects quad surfaces: "+renderer.name);
                Vector3[] vertices=mesh.vertices;
                Vector3 origin=renderer.transform.TransformPoint(vertices[0]);
                Vector3 u=renderer.transform.TransformPoint(vertices[1])-origin;
                Vector3 v=renderer.transform.TransformPoint(vertices[3])-origin;
                Vector3 normal=renderer.transform.TransformDirection(mesh.normals[0]).normalized;
                Material material=renderer.sharedMaterial;
                Color albedo=material!=null&&material.HasProperty("_BaseColor")?material.GetColor("_BaseColor"):
                    material!=null?material.color:Color.white;
                float area=Vector3.Cross(u,v).magnitude/(resolution*resolution);
                for(int i=0;i<resolution;i++)for(int j=0;j<resolution;j++)
                {
                    p.Add(origin+((i+0.5f)/resolution)*u+((j+0.5f)/resolution)*v);
                    n.Add(normal);a.Add(area);c.Add(albedo);names.Add(renderer.name);
                }
            }
            if(p.Count==0)throw new InvalidOperationException("No mesh surfaces were found in the experiment scene");
            return new SurfaceSampleSet { positions=p.ToArray(),normals=n.ToArray(),areas=a.ToArray(),albedos=c.ToArray(),names=names.ToArray() };
        }
    }
}
