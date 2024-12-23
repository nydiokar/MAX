import asyncio
from MAX.config.database_config import DatabaseConfig
from Orch.python.src.MAX.storage.ChatStorageMongoDB import MongoDBChatStorage
from MAX.types import ConversationMessage, ParticipantRole
from datetime import datetime

async def test_mongodb_connection():
    try:
        # Initialize config
        config = DatabaseConfig()
        
        # Initialize MongoDB storage
        mongo_storage = MongoDBChatStorage(
            mongo_uri=config.mongodb.uri,
            db_name=config.mongodb.database,
            collection_name=config.mongodb.chat_collection
        )
        
        # Create a test message with correct content format
        test_message = ConversationMessage(
            role=ParticipantRole.SYSTEM.value,  # Use .value for enum
            content="Test connection message",   # Direct string instead of list
            timestamp=datetime.now()
        )
        
        # Try to save the message
        success = await mongo_storage.save_chat_message(
            user_id="test_user",
            session_id="test_session",
            agent_id="test_agent",
            new_message=test_message
        )
        
        if success:
            print("Successfully connected to MongoDB and saved test message")
            
            # Try to retrieve the message
            messages = await mongo_storage.fetch_chat(
                user_id="test_user",
                session_id="test_session",
                agent_id="test_agent"
            )
            
            if messages:
                print("Successfully retrieved test message:")
                for msg in messages:
                    print(f"Role: {msg.role}")
                    print(f"Content: {msg.content}")
            else:
                print("No messages found")
                
        else:
            print("Failed to save test message")
            
    except Exception as e:
        print(f"Error connecting to MongoDB: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_mongodb_connection())