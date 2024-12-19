from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.Integer, primary_key=True)
    user_input = db.Column(db.Text, nullable=False)
    ai_response = db.Column(db.Text, nullable=False)
    search_results = db.Column(db.Text)
    raw_search_results = db.Column(db.Text)
    file_content = db.Column(db.Text)
    file_name = db.Column(db.String(255))
    file_summary = db.Column(db.Text)
    latency = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    conversation_id = db.Column(db.String(36), nullable=False)

    __table_args__ = {'extend_existing': True}

    def to_dict(self):
        return {
            'id': self.id,
            'user_input': self.user_input,
            'ai_response': self.ai_response,
            'search_results': self.search_results,
            'raw_search_results': self.raw_search_results,
            'file_name': self.file_name,
            'file_summary': self.file_summary,
            'latency': self.latency,
            'timestamp': self.timestamp.isoformat(),
            'conversation_id': self.conversation_id
        }
