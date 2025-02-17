from typing import Dict, List, Optional, Union, Any
from enum import Enum
import asyncio
from datetime import datetime, timedelta
from dataclasses import replace
from MAX.types.collaboration_types import (
    CollaborationRole,
    CollaborationMessage,
    SubTask
)
from MAX.agents import Agent
from MAX.utils.logger import Logger

class ProtocolType(Enum):
    BROADCAST = "broadcast"         # Send to all team members
    TARGETED = "targeted"          # Send to specific agents
    ROUND_ROBIN = "round_robin"    # Rotate through team members
    HIERARCHICAL = "hierarchical"  # Follow role-based hierarchy
    
class MessagePriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3

class CoordinationProtocol:
    """Manages communication protocols and synchronization between agents."""
    
    def __init__(self):
        self.message_queues: Dict[str, List[CollaborationMessage]] = {}  # agent_id -> messages
        self.active_protocols: Dict[str, ProtocolType] = {}  # task_id -> protocol
        self.sync_points: Dict[str, Dict[str, bool]] = {}  # task_id -> {agent_id -> ready}
        self.deadlines: Dict[str, datetime] = {}  # message_id -> deadline
        
    async def register_protocol(
        self,
        task_id: str,
        protocol_type: ProtocolType,
        participants: List[str]  # List of agent IDs
    ) -> None:
        """Register a new coordination protocol for a task."""
        self.active_protocols[task_id] = protocol_type
        self.message_queues.update({agent_id: [] for agent_id in participants})
        self.sync_points[task_id] = {agent_id: False for agent_id in participants}
        
    async def send_message(
        self,
        task_id: str,
        from_agent: str,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        target_roles: Optional[List[CollaborationRole]] = None,
        target_agents: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        require_response: bool = False
    ) -> str:
        """Send a message following the active protocol."""
        if task_id not in self.active_protocols:
            raise ValueError(f"No active protocol for task {task_id}")
            
        protocol = self.active_protocols[task_id]
        message_id = f"{task_id}_{datetime.utcnow().timestamp()}"
        
        message = CollaborationMessage(
            from_agent=from_agent,
            to_agent="",  # Will be set based on protocol
            message_type=protocol.value,
            content=content,
            timestamp=datetime.utcnow(),
            requires_response=require_response,
            response_deadline=datetime.utcnow() + timedelta(seconds=timeout) if timeout else None
        )
        
        if timeout:
            self.deadlines[message_id] = datetime.utcnow() + timedelta(seconds=timeout)
            
        # Handle different protocol types
        if protocol == ProtocolType.BROADCAST:
            await self._handle_broadcast(message, list(self.sync_points[task_id].keys()))
            
        elif protocol == ProtocolType.TARGETED and target_agents:
            await self._handle_targeted(message, target_agents)
            
        elif protocol == ProtocolType.ROUND_ROBIN:
            await self._handle_round_robin(task_id, message)
            
        elif protocol == ProtocolType.HIERARCHICAL and target_roles:
            await self._handle_hierarchical(message, target_roles)
            
        return message_id
        
    async def _handle_broadcast(
        self,
        message: CollaborationMessage,
        recipients: List[str]
    ) -> None:
        """Handle broadcast protocol."""
        for recipient in recipients:
            msg_copy = replace(message, to_agent=recipient)
            if recipient in self.message_queues:
                self.message_queues[recipient].append(msg_copy)
                
    async def _handle_targeted(
        self,
        message: CollaborationMessage,
        target_agents: List[str]
    ) -> None:
        """Handle targeted messaging."""
        for agent_id in target_agents:
            if agent_id in self.message_queues:
                msg_copy = replace(message, to_agent=agent_id)
                self.message_queues[agent_id].append(msg_copy)
                
    async def _handle_round_robin(
        self,
        task_id: str,
        message: CollaborationMessage
    ) -> None:
        """Handle round-robin distribution."""
        participants = list(self.sync_points[task_id].keys())
        if not participants:
            return
            
        # Find next agent in rotation
        current_idx = 0  # Could maintain last_used index for true round-robin
        next_agent = participants[current_idx]
        
        msg_copy = replace(message, to_agent=next_agent)
        self.message_queues[next_agent].append(msg_copy)
        
    async def _handle_hierarchical(
        self,
        message: CollaborationMessage,
        target_roles: List[CollaborationRole]
    ) -> None:
        """Handle hierarchical message distribution."""
        # In a real implementation, would need access to role assignments
        # For now, just demonstrate the concept
        role_hierarchy = {
            CollaborationRole.COORDINATOR: 3,
            CollaborationRole.VALIDATOR: 2,
            CollaborationRole.CONTRIBUTOR: 1,
            CollaborationRole.OBSERVER: 0
        }
        
        # Sort roles by hierarchy
        target_roles.sort(key=lambda r: role_hierarchy.get(r, 0), reverse=True)
        
        # Would need agent-role mapping to complete implementation
        pass

    async def check_sync_point(
        self,
        task_id: str,
        agent_id: str
    ) -> bool:
        """Check if all agents have reached a sync point."""
        if task_id not in self.sync_points or agent_id not in self.sync_points[task_id]:
            return False
            
        self.sync_points[task_id][agent_id] = True
        return all(self.sync_points[task_id].values())
        
    async def get_messages(
        self,
        agent_id: str,
        max_msgs: int = 10
    ) -> List[CollaborationMessage]:
        """Get pending messages for an agent."""
        if agent_id not in self.message_queues:
            return []
            
        messages = []
        while len(messages) < max_msgs and self.message_queues[agent_id]:
            msg = self.message_queues[agent_id].pop(0)
            if msg.response_deadline:
                if msg.response_deadline > datetime.utcnow():
                    messages.append(msg)
            else:
                messages.append(msg)
                
        return messages
        
    def reset_sync_point(self, task_id: str) -> None:
        """Reset sync point for a new synchronization round."""
        if task_id in self.sync_points:
            self.sync_points[task_id] = {
                agent_id: False for agent_id in self.sync_points[task_id]
            }
            
    def cleanup_task(self, task_id: str) -> None:
        """Clean up protocol resources for a task."""
        self.active_protocols.pop(task_id, None)
        self.sync_points.pop(task_id, None)
        
        # Clean up related messages
        for queue in self.message_queues.values():
            queue[:] = [msg for msg in queue if msg.subtask_id != task_id]
            
        # Clean up expired deadlines
        self.deadlines = {
            msg_id: deadline 
            for msg_id, deadline in self.deadlines.items()
            if deadline > datetime.utcnow()
        }
