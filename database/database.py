from sqlalchemy import create_engine, Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class ChatMemory(Base):
    __tablename__ = 'chat_memory'
    id = Column(Integer, primary_key=True)
    user_input = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)

engine = create_engine('sqlite:///database/chatbot.db')
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

def save_interaction(user_input, bot_response):
    interaction = ChatMemory(user_input=user_input, bot_response=bot_response)
    session.add(interaction)
    session.commit()
