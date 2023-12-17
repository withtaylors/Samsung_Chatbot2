import React, { useState, useEffect, useRef } from "react";
import "./App.css";

const App = () => {
  const [messages, setMessages] = useState([]);
  const [currentTypingId, setCurrentTypingId] = useState(null);
  const messagesListRef = useRef(null);

  const handleRefresh = () => {
    setMessages([]);
    scrollToBottom();
  };

  const handleSendMessage = async (message) => {
    setMessages((prevMessages) => [
      ...prevMessages,
      { text: message, isUser: true },
    ]);

    try {
      const response = await fetch("process_query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: message }),
      });

      const data = await response.json();

      setMessages((prevMessages) => [
        ...prevMessages,
        { text: data.response, isUser: false },
      ]);
    } catch (error) {
      console.error("Error:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: "오류가 발생했습니다. 다시 시도해주세요.", isUser: false },
      ]);
    }
  };

  const handleEndTyping = (id) => {
    setMessages((prevMessages) =>
      prevMessages.map((msg) =>
        msg.id === id ? { ...msg, isTyping: false } : msg
      )
    );
    setCurrentTypingId(null);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    if (messagesListRef.current) {
      messagesListRef.current.scrollTop = messagesListRef.current.scrollHeight;
    }
  };

  return (
    <div className="app">
      <div className="chat-box">
        <h1>삼성증권 LLM 챗봇</h1>
        <MessageList
          messages={messages}
          currentTypingId={currentTypingId}
          onEndTyping={handleEndTyping}
          messagesListRef={messagesListRef}
        />
        <MessageForm
          onSendMessage={handleSendMessage}
          onRefresh={handleRefresh} // Pass the refresh callback
        />
      </div>
    </div>
  );
};

const MessageList = ({
  messages,
  currentTypingId,
  onEndTyping,
  messagesListRef,
}) => (
  <div className="messages-list" ref={messagesListRef}>
    {messages.map((message, index) => (
      <Message
        key={index}
        {...message}
        onEndTyping={onEndTyping}
        currentTypingId={currentTypingId}
      />
    ))}
  </div>
);

const Message = ({
  text,
  isUser,
  isTyping,
  id,
  onEndTyping,
  currentTypingId,
}) => {
  return (
    <div className={isUser ? "user-message" : "ai-message"}>
      {isTyping && currentTypingId === id && text ? (
        <div speed={50} onFinishedTyping={() => onEndTyping(id)}>
          <p>
            <b>AI</b>: {text}
          </p>
        </div>
      ) : (
        <p>
          <b>{isUser ? "You " : "AI "}</b>: {text}
        </p>
      )}
    </div>
  );
};

const ErrorNotification = ({ message, onClose }) => {
  useEffect(() => {
    const timeoutId = setTimeout(onClose, 2000);

    return () => clearTimeout(timeoutId);
  }, [onClose]);

  return <div className="error-notification">{message}</div>;
};

const MessageForm = ({ onSendMessage, onRefresh }) => {
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  const handleSubmit = (event) => {
    event.preventDefault();

    if (message.trim() === "") {
      setError("질문을 입력해주세요!");
      return;
    }

    onSendMessage(message);
    setMessage("");
    setError("");
  };

  const handleRefresh = () => {
    onRefresh();
  };

  return (
    <div>
      <div>
        {error && (
          <ErrorNotification message={error} onClose={() => setError("")} />
        )}
      </div>
      <div>
        <form onSubmit={handleSubmit} className="message-form">
          <button
            type="button"
            className="refresh-button"
            onClick={handleRefresh}
          >
            ↺
          </button>
          <input
            type="text"
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            className="message-input"
          />
          <button type="submit" className="send-button">
            질문하기
          </button>
        </form>
      </div>
    </div>
  );
};

export default App;
