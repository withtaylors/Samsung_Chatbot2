import React, { useState, useEffect, useRef } from "react";
import "./App.css";

const App = () => {
  const [messages, setMessages] = useState([]);
  const [currentTypingId, setCurrentTypingId] = useState(null);
  const messagesListRef = useRef(null);

  // 선택된 그래프 이미지 이름을 저장하는 상태 변수
  const [selectedGraphImageName, setSelectedGraphImageName] = useState(null);
  const [graphImageNames, setGraphImageNames] = useState([]);

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

      // 그래프 이미지 이름들을 상태 변수에 저장
      if (data.graphImages) {
        setGraphImageNames(data.graphImages);
      }

    } catch (error) {
      console.error("Error:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: "오류가 발생했습니다. 다시 시도해주세요.", isUser: false },
      ]);
    }
  };

  const handleGraphSelection = async (imageName) => {
    // 선택된 그래프 이미지 이름 설정
    setSelectedGraphImageName(imageName);

    // 백엔드에 선택된 그래프 정보 요청
    try {
      const response = await fetch("/get_graph_description", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ graphImageName: imageName }),
      });

      const data = await response.json();

      // 받은 그래프 설명을 메시지에 추가
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: data.graphDescription, isUser: false },
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
      <div className="chat-box" style={{ whiteSpace: 'pre-wrap' }}>
        <h1>삼성증권 LLM 챗봇</h1>
        <MessageList
           messages={messages}
           currentTypingId={currentTypingId}
           onEndTyping={handleEndTyping}
           messagesListRef={messagesListRef}
           graphImageNames={graphImageNames}
           selectedGraphImageName={selectedGraphImageName}
           onGraphSelection={handleGraphSelection}
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
  graphImageNames,
  selectedGraphImageName,
  onGraphSelection
}) => (
  <div className="messages-list" ref={messagesListRef}>
    {messages.map((message, index) => (
      <Message
        key={index}
        text={message.text}
        isUser={message.isUser}
        isTyping={message.isTyping}
        id={message.id}
        onEndTyping={onEndTyping}
        currentTypingId={currentTypingId}
        graphImageNames={message.isUser ? null : graphImageNames} // AI의 메시지에만 그래프 이름을 보냄
        onGraphSelection={onGraphSelection}
      />
    ))}

    {/* 선택된 그래프 이미지 표시 */}
    {selectedGraphImageName && (
      <div className="selected-graph-container">
        <img src={`/Samsung_Chatbot2/src/그래프 png 파일/${selectedGraphImageName}.png의 사본.png`} alt="Selected Graph" className="selected-graph-image" />

      </div>
    )}
  </div>
);


const Message = ({
  text,
  isUser,
  isTyping,
  id,
  onEndTyping,
  currentTypingId,
  graphImageNames, // 그래프 이미지 이름들
  onGraphSelection, // 그래프 선택 핸들러
}) => {

  const isEmpty = (obj) => {
    return Object.keys(obj).length === 0 && obj.constructor === Object;
  };

  const messageText = typeof text === 'object' ? (isEmpty(text) ? '' : JSON.stringify(text)) : text;

  return (
    <div className={isUser ? "user-message" : "ai-message"}>
      {isTyping && currentTypingId === id && text ? (
        <div speed={50} onFinishedTyping={() => onEndTyping(id)}>
          <p>
            <b>AI</b>: {messageText}
          </p>
        </div>
      ) : (
        <>
          <p>
            <b>{isUser ? "You " : "AI "}</b>: {messageText}
          </p>
          {!isUser && graphImageNames && graphImageNames.length > 0 && 
            graphImageNames.map((imageName, index) => (
              <button key={index} onClick={() => onGraphSelection(imageName)}>
                {imageName}
              </button>
            ))
          }
        </>
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
