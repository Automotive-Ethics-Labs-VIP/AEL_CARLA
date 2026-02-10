from typing import Optional, List
from .ethical_attributes import EthicalAttributeSchema
from collections import OrderedDict
import threading
import json
from pathlib import Path
from datetime import datetime, timezone

class EthicalActorRegistry:
    def __init__(self, max_size: int = 10000) -> None:
        self._registry: OrderedDict[int, EthicalAttributeSchema] = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size

    def register(self, actor_id: int, attrs: EthicalAttributeSchema) -> None:
        with self._lock:
            if actor_id in self._registry:
                del self._registry[actor_id]

            self._registry[actor_id] = attrs

            if len(self._registry) > self._max_size:
                self._registry.popitem(last=False)

    def get_attributes(self, actor_id: int) -> Optional[EthicalAttributeSchema]:
        with self._lock:
            if actor_id not in self._registry:
                return None
        
            self._registry.move_to_end(actor_id)
            return self._registry[actor_id]
    
    def get_all_vulnerable_actors(self, threshold: float = 0.5) -> List[int]:
        with self._lock:
            return [
                actor_id 
                for actor_id, attrs in self._registry.items()
                if attrs.vulnerability_score >= threshold
            ]
    
    def export_to_json(self, filepath: str) -> None:
        with self._lock:
            snapshot = dict(self._registry)

        data = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
                "total_actors": len(snapshot),
                "carla_version": "0.9.16"
            },
            "actors": {
                str(actor_id): attrs.model_dump()
                for actor_id, attrs in snapshot.items()
            }
        }
        
        Path(filepath).write_text(json.dumps(data, indent=2))

    def import_from_json(self, filepath: str) -> None:
        data = json.loads(Path(filepath).read_text())
        
        with self._lock:
            self._registry.clear()
            
            for actor_id_str, attrs_dict in data["actors"].items():
                actor_id = int(actor_id_str)
                attrs = EthicalAttributeSchema(**attrs_dict)
                self._registry[actor_id] = attrs
                
                if len(self._registry) > self._max_size: # check max_size just in case to respect the bound
                    self._registry.popitem(last=False)