import React, { useState, useEffect, useRef } from "react";
import "./App.css";

const App = () => {
  const [messages, setMessages] = useState([]);
  const [currentTypingId, setCurrentTypingId] = useState(null);
  const messagesListRef = useRef(null);

  // 선택된 그래프 이미지 이름을 저장하는 상태 변수
  const [selectedGraphImageName, setSelectedGraphImageName] = useState(null);
  const [graphImageNames] = useState([]);

  const handleRefresh = () => {
    setMessages([]);
    setSelectedGraphImageName(null); // 선택된 그래프 이미지 이름 초기화
    scrollToBottom();
  };

  const handleSendMessage = async (message) => {
    try {
      const response = await fetch("/process_query", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: message }),
      });

      const data = await response.json();

      // 새로운 메시지 객체에 그래프 목록 추가
      const newMessage = {
        text: data.response,
        isUser: false,
        graphImageNames: message.includes("그래프") ? data.graphImages : [],
      };

      setMessages((prevMessages) => [
        ...prevMessages,
        { text: message, isUser: true },
        newMessage,
      ]);
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
      // 배열을 하나의 문자열로 변환
      const descriptionText = data.graphDescription.join(" ");
      const imageUrl = data.imageUrl; // Flask에서 받은 이미지 URL

      // 받은 그래프 설명과 선택된 이미지 이름을 메시지에 추가
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          text: `선택된 그래프 이미지: ${imageName}`,
          isUser: false,
          imageUrl: imageUrl, // Flask에서 받은 이미지 URL을 사용
        },
        { text: descriptionText, isUser: false }, // 여기서 descriptionText 사용
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
      <div className="chat-box" style={{ whiteSpace: "pre-wrap" }}>
        <h1>삼성증권 sLLM 투자정보 검색 챗봇</h1>
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
  onGraphSelection,
}) => (
  <div className="messages-list" ref={messagesListRef}>
    {messages.map((message, index) => (
      <Message
        key={index}
        text={message.text}
        isUser={message.isUser}
        imageUrl={message.imageUrl}
        isTyping={message.isTyping}
        id={message.id}
        onEndTyping={onEndTyping}
        currentTypingId={currentTypingId}
        graphImageNames={message.graphImageNames || []} // 메시지 객체의 그래프 목록을 사용
        onGraphSelection={onGraphSelection}
      />
    ))}

    {/* 선택된 그래프 이미지 표시
    {selectedGraphImageName && (
      <>
        <div>Selected Image Name: {selectedGraphImageName}</div>
        <div className="selected-graph-container">
          <img
            src={`src/[FINAL] 그래프 png 파일/${selectedGraphImageName}.png`}
            className="selected-graph-image"
          />
        </div>
      </>
    )} */}
  </div>
);

const Message = ({
  text,
  isUser,
  imageUrl,
  isTyping,
  id,
  onEndTyping,
  currentTypingId,
  graphImageNames = [], // 기본값을 빈 배열로 설정
  onGraphSelection,
}) => {
  console.log("imageUrl:", imageUrl);
  console.log("graphImageNames:", graphImageNames);

  const isEmpty = (obj) => {
    return Object.keys(obj).length === 0 && obj.constructor === Object;
  };

  const messageText =
    typeof text === "object"
      ? isEmpty(text)
        ? ""
        : JSON.stringify(text)
      : text;

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
          {imageUrl && (
            <img src={imageUrl} alt="Graph" className="message-image" />
          )}
          {!isUser && graphImageNames.length > 0 && (
            <div className="graph-list">
              {graphImageNames.map((imageName, index) => (
                <button
                  key={`${imageName}-${index}`} // 고유한 키를 생성
                  onClick={() =>
                    onGraphSelection && onGraphSelection(imageName)
                  }
                  className="graph-list-item"
                >
                  {index + 1}. {imageName}
                </button>
              ))}
            </div>
          )}
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
