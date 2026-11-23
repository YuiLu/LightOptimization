using System;
using System.Collections.Generic;
using UnityEngine;

namespace LightingMvp
{
    /// <summary>World-space, non-overlapping surface patch samples read from scene assets.</summary>
    public sealed class SurfaceSampleSet
    {
        public Vector3[] positions,normals;
        public float[] areas;
        public Color[] albedos;
        public string[] names;
        internal Patch[] patches;
        public int Count { get { return positions.Length; } }

        internal readonly struct Patch
        {
            private readonly Vector3 a,b,c,d;
            private readonly bool quad;

            public Patch(Vector3 a,Vector3 b,Vector3 c,Vector3 d=default,bool quad=false)
            { this.a=a;this.b=b;this.c=c;this.d=d;this.quad=quad; }

            public Vector3[] Quadrature(int order)
            {
                var points=new Vector3[order*order];
                int k=0;
                if(quad)
                {
                    for(int i=0;i<order;i++)for(int j=0;j<order;j++)
                        points[k++]=a+((i+0.5f)/order)*(b-a)+((j+0.5f)/order)*(d-a);
                }
                else AddTrianglePoints(a,b,c,order);
                return points;

                void AddTrianglePoints(Vector3 v0,Vector3 v1,Vector3 v2,int size)
                {
                    if(size==1){points[k++]=(v0+v1+v2)/3;return;}
                    Vector3 ab=(v0+v1)*0.5f,bc=(v1+v2)*0.5f,ca=(v2+v0)*0.5f;
                    int half=size/2;
                    AddTrianglePoints(v0,ab,ca,half);AddTrianglePoints(ab,v1,bc,half);
                    AddTrianglePoints(ca,bc,v2,half);AddTrianglePoints(ab,bc,ca,half);
                }
            }
        }

        public static SurfaceSampleSet FromRenderers(IEnumerable<MeshRenderer> renderers,int resolution)
        {
            if(resolution<1)throw new ArgumentOutOfRangeException("resolution");
            var p=new List<Vector3>();var n=new List<Vector3>();var a=new List<float>();var c=new List<Color>();var names=new List<string>();var patches=new List<Patch>();
            foreach(var renderer in renderers)
            {
                var filter=renderer.GetComponent<MeshFilter>();Mesh mesh=filter!=null?filter.sharedMesh:null;if(mesh==null)continue;
                Color albedo=DiffuseAlbedo(renderer);
                Matrix4x4 normalMatrix=renderer.transform.worldToLocalMatrix.transpose;
                Vector3[] vertices=mesh.vertices;Vector3[] normals=mesh.normals;int[] triangles=mesh.triangles;
                if(vertices.Length==4&&triangles.Length==6&&normals.Length==4&&
                    (vertices[0]+vertices[2]-vertices[1]-vertices[3]).sqrMagnitude<1e-10f)
                {
                    Vector3 origin=renderer.transform.TransformPoint(vertices[0]);
                    Vector3 u=renderer.transform.TransformPoint(vertices[1])-origin,v=renderer.transform.TransformPoint(vertices[3])-origin;
                    Vector3 normal=normalMatrix.MultiplyVector(normals[0]).normalized;
                    AddQuad(origin,u,v,normal,albedo,renderer.name,resolution,p,n,a,c,names,patches);
                }
                else
                {
                    for(int t=0;t<triangles.Length;t+=3)
                    {
                        int ia=triangles[t],ib=triangles[t+1],ic=triangles[t+2];
                        Vector3 va=renderer.transform.TransformPoint(vertices[ia]),vb=renderer.transform.TransformPoint(vertices[ib]),vc=renderer.transform.TransformPoint(vertices[ic]);
                        Vector3 face=Vector3.Cross(vb-va,vc-va);
                        if(face.sqrMagnitude<1e-20f)continue;
                        bool hasNormals=normals.Length==vertices.Length;
                        Vector3 na=hasNormals?normalMatrix.MultiplyVector(normals[ia]):face;
                        Vector3 nb=hasNormals?normalMatrix.MultiplyVector(normals[ib]):face;
                        Vector3 nc=hasNormals?normalMatrix.MultiplyVector(normals[ic]):face;
                        AddTriangle(va,vb,vc,na,nb,nc,albedo,renderer.name,resolution,p,n,a,c,names,patches);
                    }
                }
            }
            if(p.Count==0)throw new InvalidOperationException("No mesh surfaces were found in the experiment scene");
            return new SurfaceSampleSet { positions=p.ToArray(),normals=n.ToArray(),areas=a.ToArray(),albedos=c.ToArray(),names=names.ToArray(),patches=patches.ToArray() };
        }

        public static Color DiffuseAlbedo(MeshRenderer renderer)
        {
            Material material=renderer.sharedMaterial;
            if(material==null)return Color.white;
            if(material.HasProperty("_BaseColor"))return material.GetColor("_BaseColor");
            return material.HasProperty("_Color")?material.GetColor("_Color"):Color.white;
        }

        private static void AddQuad(Vector3 origin,Vector3 u,Vector3 v,Vector3 normal,Color albedo,string name,int resolution,
            List<Vector3> p,List<Vector3> n,List<float> a,List<Color> c,List<string> names,List<Patch> patches)
        {
            float area=Vector3.Cross(u,v).magnitude/(resolution*resolution);
            for(int i=0;i<resolution;i++)for(int j=0;j<resolution;j++)
            {
                p.Add(origin+((i+0.5f)/resolution)*u+((j+0.5f)/resolution)*v);n.Add(normal);a.Add(area);c.Add(albedo);names.Add(name);
                Vector3 corner=origin+i*(u/resolution)+j*(v/resolution);
                patches.Add(new Patch(corner,corner+u/resolution,corner+(u+v)/resolution,corner+v/resolution,true));
            }
        }

        private static void AddTriangle(Vector3 va,Vector3 vb,Vector3 vc,Vector3 na,Vector3 nb,Vector3 nc,Color albedo,string name,int resolution,
            List<Vector3> p,List<Vector3> n,List<float> a,List<Color> c,List<string> names,List<Patch> patches)
        {
            float area=Vector3.Cross(vb-va,vc-va).magnitude*0.5f/(resolution*resolution);
            for(int i=0;i<resolution;i++)for(int j=0;j<resolution-i;j++)
            {
                AddBarycentric((i+1f/3f)/resolution,(j+1f/3f)/resolution);
                patches.Add(new Patch(Vertex(i,j),Vertex(i+1,j),Vertex(i,j+1)));
                if(i+j<resolution-1)
                {
                    AddBarycentric((i+2f/3f)/resolution,(j+2f/3f)/resolution);
                    patches.Add(new Patch(Vertex(i+1,j+1),Vertex(i,j+1),Vertex(i+1,j)));
                }
            }
            Vector3 Vertex(int i,int j)=>va+(i/(float)resolution)*(vb-va)+(j/(float)resolution)*(vc-va);
            void AddBarycentric(float u,float v)
            {
                p.Add(va+u*(vb-va)+v*(vc-va));n.Add((na+u*(nb-na)+v*(nc-na)).normalized);a.Add(area);c.Add(albedo);names.Add(name);
            }
        }
    }
}
