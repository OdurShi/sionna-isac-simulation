import os
import sionna
import open3d as o3d

def view_model(ply_path):
    """通用的 3D 模型加载与渲染函数"""
    print(f"⚙️ 正在使用 Open3D 加载网格数据...\n文件路径: {ply_path}")
    mesh = o3d.io.read_triangle_mesh(ply_path)
    
    if mesh.is_empty():
        print("❌ 模型加载失败或为空！")
        return
        
    # 计算顶点法线，让建筑物在光照下产生真实的立体阴影
    mesh.compute_vertex_normals() 
    
    print("🚀 正在弹出 3D 交互查看器...")
    print("-" * 30)
    print("💡 【操作指南】")
    print("   👉 鼠标左键拖拽：全视角旋转")
    print("   👉 Shift + 鼠标左键拖拽：平移地图")
    print("   👉 鼠标滚轮：拉近/推远")
    print("   👉 按 'Q' 键或直接点红叉：退出查看器")
    print("-" * 30)
    
    # 唤醒本地图形界面进行渲染
    o3d.visualization.draw_geometries([mesh], 
                                      window_name="Sionna 3D City Viewer",
                                      width=1280, height=720,
                                      left=100, top=100)

def main():
    print("🎬 正在定位 Sionna 内置的巴黎星形广场 3D 模型...")
    
    # 获取 Sionna 底层巴黎场景的 XML 绝对路径
    paris_xml_path = sionna.rt.scene.etoile
    
    # 获取该 XML 文件所在的目录路径
    scene_dir = os.path.dirname(paris_xml_path)
    print(f"🔍 正在深入目录搜寻真正的 3D 文件: {scene_dir}")
    
    # 智能搜索机制：遍历该目录及其所有子目录，寻找 .ply 或 .obj 文件
    target_mesh = None
    for root, dirs, files in os.walk(scene_dir):
        for file in files:
            # 找到第一个支持的 3D 格式文件就抓取它
            if file.endswith('.ply') or file.endswith('.obj'):
                target_mesh = os.path.join(root, file)
                break
        if target_mesh:
            break
            
    if target_mesh:
        print(f"✅ 抓到隐藏的模型文件啦：{target_mesh}")
        view_model(target_mesh)
    else:
        print("❌ 见鬼了，在这个目录里翻了个底朝天也没找到 3D 源文件！")

if __name__ == "__main__":
    main()