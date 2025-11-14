import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  PaperAirplaneIcon,
  SparklesIcon,
  BeakerIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import axios from 'axios'
import MessageBubble from './MessageBubble'
import TypingIndicator from './TypingIndicator'

const ChatInterface = () => {
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
  const [analysisData, setAnalysisData] = useState(null)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

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
      const response = await axios.post('/api/rag-chat/', {
        message: inputMessage
      })

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.data.message,
        timestamp: new Date().toISOString(),
        analysis: response.data.full_analysis
      }

      setMessages(prev => [...prev, botMessage])
      setAnalysisData(response.data.full_analysis)
      
    } catch (error) {
      console.error('Error sending message:', error)
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

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const testFeatures = async () => {
    setIsLoading(true)
    try {
      const response = await axios.get('http://127.0.0.1:8000/api/test-all-features/')
      const testMessage = {
        id: Date.now(),
        type: 'bot',
        content: `🧪 Feature Test Results:\n\n${response.data.test_results.summary.tests_passed}/${response.data.test_results.summary.total_tests} tests passed (${response.data.test_results.summary.success_rate})`,
        timestamp: new Date().toISOString(),
        analysis: response.data.test_results
      }
      setMessages(prev => [...prev, testMessage])
    } catch (error) {
      console.error('Error testing features:', error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="card-premium overflow-hidden h-[600px] flex flex-col"
    >
      {/* Chat Header */}
      <div className="bg-gradient-to-r from-primary-500 to-emerald-500 p-6 text-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <motion.div
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center"
            >
              <SparklesIcon className="w-5 h-5" />
            </motion.div>
            <div>
              <h2 className="text-xl font-bold">RAG Chatbot</h2>
              <p className="text-primary-100 text-sm">AI-Powered Clinical Assistant</p>
            </div>
          </div>
          
          <motion.button
            onClick={testFeatures}
            disabled={isLoading}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center space-x-2 px-4 py-2 bg-white/20 hover:bg-white/30 rounded-xl transition-all duration-200 disabled:opacity-50"
          >
            <BeakerIcon className="w-4 h-4" />
            <span className="text-sm font-medium">Test Features</span>
          </motion.button>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gradient-to-b from-white/50 to-primary-50/30 scrollbar-hide">
        <AnimatePresence>
          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}
        </AnimatePresence>
        
        {isLoading && <TypingIndicator />}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="p-6 bg-white/60 backdrop-blur-sm border-t border-primary-200/30">
        <div className="flex space-x-4">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your medical query here..."
              className="w-full px-6 py-4 bg-white/80 border-2 border-primary-200 focus:border-primary-500 focus:ring-4 focus:ring-primary-200/50 rounded-2xl outline-none transition-all duration-300 resize-none overflow-hidden placeholder-neutral-500 font-medium"
              rows="1"
              style={{ minHeight: '56px', maxHeight: '120px' }}
              disabled={isLoading}
            />
          </div>
          
          <motion.button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="w-14 h-14 bg-gradient-to-r from-primary-500 to-emerald-500 text-white rounded-2xl flex items-center justify-center shadow-medical hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 group"
          >
            <PaperAirplaneIcon className="w-6 h-6 group-hover:translate-x-0.5 transition-transform duration-200" />
          </motion.button>
        </div>
        
        {/* Quick Actions */}
        <div className="flex flex-wrap gap-2 mt-4">
          {[
            "35 year old cancer patient having headache problem",
            "Chest pain and shortness of breath",
            "High blood pressure symptoms"
          ].map((suggestion, index) => (
            <motion.button
              key={index}
              onClick={() => setInputMessage(suggestion)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="px-3 py-2 text-xs bg-white/60 hover:bg-white/80 text-neutral-600 hover:text-neutral-800 rounded-lg border border-primary-200/50 hover:border-primary-300 transition-all duration-200 font-medium"
            >
              {suggestion}
            </motion.button>
          ))}
        </div>
      </div>
    </motion.div>
  )
}

export default ChatInterface
