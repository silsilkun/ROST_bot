from __future__ import annotations

from typing import List, Tuple
import math

import rclpy
from rclpy.node import Node

from rost_interfaces.srv import perception_to_control




def validate_objects_flat(data: List[float]) -> Tuple[bool, str]:
    if data is None:
        return False, "data is None"
    if len(data) == 0:
        return True, "empty (N=0)" #빈 리스트는 허용
    if len(data) % 5 != 0:
        return False, f"len must be multiple of 5, got {len(data)}"
    for v in data:
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return False, "NaN/Inf found"
        return True, "ok"
    
    
class perception_server(Node):
    """
    Control이 요청하면 현재 보유한 objects_flat([id,x,y,z,angle]*N)을 응답으로 반환하는 서버.
    내부 데이터는 너의 perception 파이프라인 결과로 업데이트해주면 됨.
    """
    
    def __init__(self):
        super().__init__("perception_server")
        
        self.declare_parameter("perception_to_control", "/perception_to_control")
        perception_to_control = self.get_parameter("perception_to_control").value
        
        #최신 결과 캐시 (perception 루프에서 update)
        self._objects_flat: List[float] = []
        
        self._srv =self.create_service(perception_to_control, perception_to_control, self.on_request)
        
        self.get_logger().info(f"[PerceptionServer] Ready: {perception_to_control}")
        
    # -------------------------
    # 외부(perception)에서 호출해서 최신 결과 갱신
    # -------------------------
    def update_objects(self, objects_flat: List[float]) -> None:
        ok, reason = validate_objects_flat(objects_flat)
        if not ok:
            self.get_logger().warn(f"[PerceptionServer] update_objects: invalid data: {reason} -> ignore")
            return
        self._objects_flat = list(objects_flat)
        
    # -------------------------
    # Service callback
    # -------------------------
    def _on_request(self, request: perception_to_control.Request,
                    response: perception_to_control.Response):
        # request.request가 False여도 응답은 주되, 규칙을 정해도 됨.
        data = list(self._objects_flat)
        
        ok, reason = validate_objects_flat(data)
        if not ok:
            response.objects_flat = []
            response.objects_count = 0
            response.success = True
            response.message = "ok"
            return response
        
        
def main(args=None):
    rclpy.init(args=args)
    node = perception_server()
    
    # ✅ 예시: 테스트용 더미 데이터 주입 (실제론 perception 결과로 update_objects 호출)
    # [id,x,y,z,angle] * N
    node.update_objects([
        1.0, 0.10, 0.20, 0.30, 15.0,
        2.0, 0.11, 0.22, 0.33, 30.0,
    ])
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        
        
if __name__ == "__main__":
    main()