import { useState } from 'react'
import './App.css'

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: "Hello! I'm your RAG-enhanced Clinical Decision Support System. How can I help you today?",
      timestamp: new Date().toISOString()
    }
  ])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)

    try {
      const response = await fetch('/api/rag-chat/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputMessage })
      })

      if (response.ok) {
        const data = await response.json()
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: data.message,
          timestamp: new Date().toISOString()
        }
        setMessages(prev => [...prev, botMessage])
      } else {
        throw new Error('Failed to get response')
      }
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: `❌ Connection failed. Please check if the Django server is running on http://127.0.0.1:8000/`,
        timestamp: new Date().toISOString(),
        isError: true
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 via-white to-emerald-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-green-600 to-emerald-600 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center justify-center space-x-4">
            <div className="w-12 h-12 bg-white/20 rounded-2xl flex items-center justify-center">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/>
              </svg>
            </div>
            <div>
              <h1 className="text-3xl font-bold">CDSS Platform</h1>
              <p className="text-green-100">Clinical Decision Support System</p>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-6xl mx-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Chat Interface */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-3xl shadow-xl overflow-hidden">
              
              {/* Chat Header */}
              <div className="bg-gradient-to-r from-green-500 to-emerald-500 p-6 text-white">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
                    </svg>
                  </div>
                  <div>
                    <h2 className="text-xl font-bold">RAG Chatbot</h2>
                    <p className="text-green-100 text-sm">AI-Powered Clinical Assistant</p>
                  </div>
                </div>
              </div>

              {/* Messages Area */}
              <div className="h-96 overflow-y-auto p-6 space-y-4 bg-gradient-to-b from-white to-green-50">
                {messages.map((message) => (
                  <div key={message.id} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                    <div className={`max-w-md px-6 py-4 rounded-2xl shadow-md ${
                      message.type === 'user'
                        ? 'bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-br-md'
                        : message.isError
                          ? 'bg-red-50 text-red-800 border border-red-200 rounded-bl-md'
                          : 'bg-white text-gray-800 border border-gray-200 rounded-bl-md'
                    }`}>
                      <p className="font-medium leading-relaxed">{message.content}</p>
                      <div className={`text-xs mt-2 text-right ${
                        message.type === 'user' ? 'text-green-100' : 'text-gray-500'
                      }`}>
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                ))}
                
                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-white border border-gray-200 px-6 py-4 rounded-2xl rounded-bl-md shadow-md">
                      <div className="flex items-center space-x-2">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                          <div className="w-2 h-2 bg-green-300 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                        </div>
                        <span className="text-gray-600 font-medium">AI is thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Input Area */}
              <div className="p-6 bg-white/60 border-t border-gray-200">
                <div className="flex space-x-4">
                  <input
                    type="text"
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                    placeholder="Type your medical query here..."
                    className="flex-1 px-6 py-4 border-2 border-gray-300 focus:border-green-500 focus:ring-4 focus:ring-green-200 rounded-2xl outline-none transition-all duration-300 font-medium"
                    disabled={isLoading}
                  />
                  <button
                    onClick={handleSendMessage}
                    disabled={!inputMessage.trim() || isLoading}
                    className="w-14 h-14 bg-gradient-to-r from-green-500 to-emerald-500 text-white rounded-2xl flex items-center justify-center shadow-lg hover:shadow-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300"
                  >
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Analysis Panel */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-3xl shadow-xl overflow-hidden">
              
              {/* Panel Header */}
              <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-6 text-white">
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
                    </svg>
                  </div>
                  <div>
                    <h2 className="text-xl font-bold">Analysis Panel</h2>
                    <p className="text-blue-100 text-sm">AI Processing Results</p>
                  </div>
                </div>
              </div>

              {/* Panel Content */}
              <div className="h-96 overflow-y-auto p-6 bg-gradient-to-b from-white to-blue-50">
                <div className="bg-white/60 rounded-2xl p-6 border border-gray-200 shadow-sm">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                    <h3 className="text-lg font-bold text-green-700">Welcome</h3>
                  </div>
                  <p className="text-gray-700">Detailed analysis of your query will appear here after you send a message.</p>
                </div>
              </div>
            </div>
          </div>
          
        </div>

        {/* Features Section */}
        <div className="mt-16 text-center">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Advanced Clinical Intelligence
          </h2>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto mb-12">
            Empowering healthcare professionals with AI-driven insights for better patient outcomes
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              {
                title: 'AI-Powered Analysis',
                description: 'Advanced machine learning for clinical decision support',
                color: 'from-purple-500 to-pink-500'
              },
              {
                title: 'Patient-Centered Care',
                description: 'Personalized recommendations based on individual patient data',
                color: 'from-red-500 to-rose-500'
              },
              {
                title: 'Evidence-Based Medicine',
                description: 'Backed by the latest medical research and guidelines',
                color: 'from-green-500 to-emerald-500'
              }
            ].map((feature, index) => (
              <div key={index} className="bg-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-300">
                <div className={`w-12 h-12 bg-gradient-to-r ${feature.color} rounded-xl flex items-center justify-center text-white shadow-lg mb-4`}>
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                  </svg>
                </div>
                <h3 className="text-lg font-bold text-gray-900 mb-2">{feature.title}</h3>
                <p className="text-gray-600 text-sm leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
