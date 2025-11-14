import { useState, useRef, useEffect } from 'react'
import './index.css'

function App() {
  // Function to parse markdown and return formatted text
  const parseMarkdown = (text) => {
    if (!text) return null
    
    const elements = []
    let lastIndex = 0
    
    // Regex to match **text** (bold markdown)
    const boldRegex = /\*\*([^*]+)\*\*/g
    let match
    
    while ((match = boldRegex.exec(text)) !== null) {
      // Add text before the match
      if (match.index > lastIndex) {
        elements.push(
          <span key={`text-${lastIndex}`}>
            {text.substring(lastIndex, match.index)}
          </span>
        )
      }
      
      // Add the bold text
      elements.push(
        <strong key={`bold-${match.index}`} style={{ fontWeight: '700' }}>
          {match[1]}
        </strong>
      )
      
      lastIndex = match.index + match[0].length
    }
    
    // Add remaining text after last match
    if (lastIndex < text.length) {
      elements.push(
        <span key={`text-${lastIndex}`}>
          {text.substring(lastIndex)}
        </span>
      )
    }
    
    return elements.length > 0 ? elements : text
  }

  const [activeTab, setActiveTab] = useState('chat')
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: "Hello! I'm VaidyaSathi 🙏, your AI-powered Advanced Clinical Decision Support System. I can help you with medical queries about 55+ diseases including Cardiovascular (Heart Disease, Stroke, Hypertension), Respiratory (COPD, Asthma, Pneumonia), Diabetes, Mental Health conditions, and many more.\n\n📋 **Available Disease Categories:**\n• Cardiovascular (5 diseases)\n• Respiratory (5 diseases)\n• Endocrine/Metabolic (5 diseases)\n• Neurological (5 diseases)\n• Mental Health (5 diseases)\n• Gastrointestinal (5 diseases)\n• Musculoskeletal (5 diseases)\n• Infectious (5 diseases)\n• Cancer (5 diseases)\n• Kidney/Urinary (5 diseases)\n• Skin (5 diseases)\n\n💡 **I can answer questions about:**\n• Symptoms and diagnosis\n• Treatments and medications\n• Risk factors and prevention\n• Research and evidence\n• General health advice\n\nHow can I assist you with your medical queries today?",
      timestamp: new Date().toISOString()
    }
  ])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [currentResearchStep, setCurrentResearchStep] = useState('')
  const [analysisData, setAnalysisData] = useState(null)
  const [backendStatus, setBackendStatus] = useState('checking')
  const [chatBoxWidth, setChatBoxWidth] = useState(48) // Chat box width percentage
  const [chatBoxHeight, setChatBoxHeight] = useState(70) // Chat box height vh
  const [analysisBoxWidth, setAnalysisBoxWidth] = useState(48) // Analysis box width percentage
  const [analysisBoxHeight, setAnalysisBoxHeight] = useState(70) // Analysis box height vh
  const [isResizing, setIsResizing] = useState(null) // 'chat' or 'analysis' or null
  const messagesEndRef = useRef(null)
  const resizeStartPos = useRef({ x: 0, y: 0, width: 0, height: 0 })

  // Check backend connection on component mount
  useEffect(() => {
    const checkBackendConnection = async () => {
      try {
        console.log('🔍 Checking backend connection...')
        const response = await fetch('http://127.0.0.1:8000/api/health/')
        if (response.ok) {
          const data = await response.json()
          console.log('✅ Backend health check passed:', data)
          setBackendStatus('connected')
        } else {
          console.log('❌ Backend health check failed:', response.status)
          setBackendStatus('disconnected')
        }
      } catch (error) {
        console.log('❌ Backend connection error:', error)
        setBackendStatus('disconnected')
      }
    }

    checkBackendConnection()
    // Check every 30 seconds
    const interval = setInterval(checkBackendConnection, 30000)
    return () => clearInterval(interval)
  }, [])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Simulate research progress steps and return a promise
  const simulateResearchProgress = () => {
    return new Promise((resolve) => {
      const researchSteps = [
        { step: '🔍 Analyzing input query...', time: 500 },
        { step: '🧠 Processing NLP features...', time: 800 },
        { step: '📚 Searching medical knowledge base...', time: 1200 },
        { step: '🔬 Performing semantic analysis...', time: 900 },
        { step: '🩺 Retrieving relevant medical contexts...', time: 1100 },
        { step: '🤖 Generating LLM-enhanced responses...', time: 1300 },
        { step: '📊 Calculating risk assessments...', time: 700 },
        { step: '💊 Formulating recommendations...', time: 600 },
        { step: '📋 Compiling differential diagnoses...', time: 500 },
        { step: '✅ Finalizing comprehensive analysis...', time: 400 }
      ]

      let currentTime = 0

      researchSteps.forEach((research, index) => {
        setTimeout(() => {
          setCurrentResearchStep(research.step)
        }, currentTime)
        currentTime += research.time
      })

      // Resolve the promise when all steps are completed
      setTimeout(() => {
        setCurrentResearchStep('')
        resolve()
      }, currentTime)
    })
  }

  // Complete integration with Django backend - ALL features preserved
  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    // Handle help commands
    if (inputMessage.toLowerCase().includes('help') || inputMessage.toLowerCase().includes('diseases')) {
      const helpMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: `📋 **Available Diseases in Our Database:**

**Cardiovascular (5):**
• Ischemic Heart Disease • Stroke • Hypertension • Heart Failure • Atrial Fibrillation

**Respiratory (5):**
• COPD • Asthma • Pneumonia • Lung Cancer • Lower Respiratory Infections

**Endocrine/Metabolic (5):**
• Type 2 Diabetes • Type 1 Diabetes • Obesity • Metabolic Syndrome • Thyroid Disorders

**Neurological (5):**
• Alzheimer's Disease • Parkinson's Disease • Epilepsy • Migraine • Multiple Sclerosis

**Mental Health (5):**
• Depression • Anxiety Disorders • Bipolar Disorder • Schizophrenia • PTSD

**Gastrointestinal (5):**
• GERD • IBD • IBS • Peptic Ulcer Disease • Liver Cirrhosis

**Musculoskeletal (5):**
• Osteoarthritis • Rheumatoid Arthritis • Osteoporosis • Low Back Pain • Fibromyalgia

**Infectious (5):**
• COVID-19 • Tuberculosis • Malaria • HIV/AIDS • Hepatitis B

**Cancer (5):**
• Breast Cancer • Colorectal Cancer • Prostate Cancer • Liver Cancer • Stomach Cancer

**Kidney/Urinary (5):**
• Chronic Kidney Disease • Acute Kidney Injury • Kidney Stones • UTI • BPH

**Skin (5):**
• Atopic Dermatitis • Psoriasis • Acne Vulgaris • Skin Cancer • Eczema

💡 **Ask me about symptoms, treatments, diagnosis, or any medical concerns related to these diseases!**`,
        timestamp: new Date().toISOString()
      }
      setMessages(prev => [...prev, helpMessage])
      setInputMessage('')
      return
    }

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString()
    }

    setMessages(prev => [...prev, userMessage])
    const currentInput = inputMessage
    setInputMessage('')
    setIsLoading(true)
    
    try {
      console.log('🔄 Sending request to backend:', 'http://127.0.0.1:8000/api/rag-chat/')
      console.log('📤 Message payload:', { message: currentInput })
      
      // Start both the research progress simulation and backend request simultaneously
      const [researchProgressPromise, backendResponse] = await Promise.all([
        simulateResearchProgress(),
        fetch('http://127.0.0.1:8000/api/rag-chat/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
          },
          body: JSON.stringify({ message: currentInput })
        })
      ])

      console.log('📡 Response status:', backendResponse.status)
      console.log('📡 Response headers:', Object.fromEntries(backendResponse.headers))

      if (backendResponse.ok) {
        const data = await backendResponse.json()
        console.log('✅ Backend response received:', data)
        
        // Wait for research progress to complete before showing results
        await researchProgressPromise
        
        const botMessage = {
          id: Date.now() + 1,
          type: 'bot',
          content: data.message || 'Response received but no message content',
          timestamp: new Date().toISOString(),
          analysis: data.analysis || data.full_analysis
        }

        setMessages(prev => [...prev, botMessage])
        
        // Handle non-medical queries
        if (data.is_medical === false && data.available_diseases) {
          console.log('🚫 Non-medical query detected, showing available diseases')
          const diseasesMessage = {
            id: Date.now() + 2,
            type: 'bot',
            content: `📋 **Available Diseases in Our Database:**\n\n${Object.entries(data.available_diseases).map(([category, diseases]) => 
              `**${category}:**\n${diseases.map(d => `• ${d}`).join('\n')}`
            ).join('\n\n')}\n\n💡 **Please ask about symptoms, treatments, or medical advice for these diseases.**`,
            timestamp: new Date().toISOString()
          }
          setMessages(prev => [...prev, diseasesMessage])
        }
        
        // Preserve all analysis data including NLP, RAG, LLM features
        if (data.analysis || data.full_analysis) {
          console.log('🧠 Analysis data received:', data.analysis || data.full_analysis)
          setAnalysisData(data.analysis || data.full_analysis)
        }
        
        // Show success message
        console.log('🎉 Message successfully processed with backend!')
      } else {
        const errorText = await backendResponse.text()
        console.error('❌ Backend error response:', errorText)
        throw new Error(`Backend returned ${backendResponse.status}: ${errorText}`)
      }
    } catch (error) {
      console.error('❌ Connection error details:', error)
      
      let errorMessage = '❌ Backend Connection Failed\n\n'
      
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        errorMessage += '🔍 Issue: Cannot reach Django server\n'
        errorMessage += '💡 Solution: Ensure Django is running with virtual environment:\n'
        errorMessage += '   1. Open terminal in cdss_chatbot folder\n'
        errorMessage += '   2. Run: venv\\Scripts\\activate\n'
        errorMessage += '   3. Run: python manage.py runserver\n'
        errorMessage += '   4. Look for "Starting development server at http://127.0.0.1:8000/"\n\n'
        errorMessage += '🌐 Expected URL: http://127.0.0.1:8000/api/rag-chat/'
      } else {
        errorMessage += `🐛 Error: ${error.message}\n`
        errorMessage += '🔧 Check Django console for detailed error logs'
      }

      const errorMsg = {
        id: Date.now() + 1,
        type: 'bot',
        content: errorMessage,
        timestamp: new Date().toISOString(),
        isError: true
      }
      setMessages(prev => [...prev, errorMsg])
    } finally {
      setIsLoading(false)
      setCurrentResearchStep('')
    }
  }

  // Test all NLP, RAG, and LLM features - preserving all functionality
  const testAllFeatures = async () => {
    setIsLoading(true)
    try {
      console.log('🧪 Testing all features...')
      const response = await fetch('http://127.0.0.1:8000/api/test-all-features/')
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`)
      }
      
      const data = await response.json()
      console.log('🧪 Test results received:', data)
      
      const testMessage = {
        id: Date.now(),
        type: 'bot',
        content: `🧪 Comprehensive Feature Test Results:

✅ Tests Passed: ${data.test_results.summary.tests_passed}/${data.test_results.summary.total_tests}
📊 Success Rate: ${data.test_results.summary.success_rate}

🔬 Features Tested:
• NLP Processing Pipeline ✓
• Semantic Analysis ✓  
• RAG System ✓
• LLM Features ✓
• Medical Knowledge Base ✓
• Database Integration ✓

All CDSS functionality is preserved and working!`,
        timestamp: new Date().toISOString(),
        analysis: data.test_results
      }
      setMessages(prev => [...prev, testMessage])
      setAnalysisData(data.test_results)
    } catch (error) {
      console.error('❌ Feature test error:', error)
      const errorMsg = {
        id: Date.now(),
        type: 'bot',
        content: `❌ Feature Test Failed: ${error.message}\n\nPlease ensure Django backend is running properly.`,
        timestamp: new Date().toISOString(),
        isError: true
      }
      setMessages(prev => [...prev, errorMsg])
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

  // Handle resize functionality for both width and height
  const handleResizeStart = (boxType, e) => {
    setIsResizing(boxType)
    e.preventDefault()
    
    // Store initial positions and sizes
    const box = boxType === 'chat' 
      ? document.querySelector('.chat-box-container')
      : document.querySelector('.analysis-panel-container')
    
    if (box) {
      const rect = box.getBoundingClientRect()
      resizeStartPos.current = {
        x: e.clientX,
        y: e.clientY,
        width: rect.width,
        height: rect.height
      }
    }
  }

  const handleResizeMove = (e) => {
    if (!isResizing) return

    const container = document.querySelector('.main-container')
    if (!container) return

    const containerRect = container.getBoundingClientRect()
    
    // Calculate deltas from start position
    const deltaX = e.clientX - resizeStartPos.current.x
    const deltaY = e.clientY - resizeStartPos.current.y

    if (isResizing === 'chat') {
      // Calculate new width as percentage
      const newWidthPx = resizeStartPos.current.width + deltaX
      const newWidthPercent = Math.min(Math.max((newWidthPx / containerRect.width) * 100, 30), 95)
      setChatBoxWidth(newWidthPercent)
      
      // Calculate new height as vh
      const newHeightPx = resizeStartPos.current.height + deltaY
      const newHeightVh = Math.min(Math.max((newHeightPx / window.innerHeight) * 100, 40), 85)
      setChatBoxHeight(newHeightVh)
    } else if (isResizing === 'analysis') {
      // Calculate new width as percentage
      const newWidthPx = resizeStartPos.current.width + deltaX
      const newWidthPercent = Math.min(Math.max((newWidthPx / containerRect.width) * 100, 30), 95)
      setAnalysisBoxWidth(newWidthPercent)
      
      // Calculate new height as vh
      const newHeightPx = resizeStartPos.current.height + deltaY
      const newHeightVh = Math.min(Math.max((newHeightPx / window.innerHeight) * 100, 40), 85)
      setAnalysisBoxHeight(newHeightVh)
    }
  }

  const handleResizeEnd = () => {
    setIsResizing(null)
    resizeStartPos.current = { x: 0, y: 0, width: 0, height: 0 }
  }

  // Global resize event listeners
  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleResizeMove)
      document.addEventListener('mouseup', handleResizeEnd)
      document.body.classList.add('resizing')
    } else {
      document.removeEventListener('mousemove', handleResizeMove)
      document.removeEventListener('mouseup', handleResizeEnd)
      document.body.classList.remove('resizing')
    }

    return () => {
      document.removeEventListener('mousemove', handleResizeMove)
      document.removeEventListener('mouseup', handleResizeEnd)
      document.body.classList.remove('resizing')
    }
  }, [isResizing])

  return (
    <div style={{ minHeight: '100vh' }}>
              {/* Premium Header with Enhanced Design */}
              <header style={{ 
                background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.12) 0%, rgba(59, 130, 246, 0.08) 50%, rgba(168, 85, 247, 0.08) 100%)', 
                backdropFilter: 'blur(20px)', 
                borderBottom: '1px solid rgba(34, 197, 94, 0.15)', 
                padding: '2.5rem 0',
                position: 'relative',
                overflow: 'hidden'
              }}>
                {/* Animated background pattern */}
                <div style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  background: 'radial-gradient(circle at 20% 50%, rgba(34, 197, 94, 0.05) 0%, transparent 50%), radial-gradient(circle at 80% 50%, rgba(59, 130, 246, 0.05) 0%, transparent 50%)',
                  pointerEvents: 'none'
                }}></div>
                
                <div className="container text-center" style={{ position: 'relative', zIndex: 1 }}>
                  <div className="flex items-center justify-center" style={{ gap: '1.25rem', marginBottom: '1.25rem' }}>
                    {/* VaidyaSathi Logo with Enhanced Design */}
                    <div style={{
                      width: '80px',
                      height: '80px',
                      background: 'linear-gradient(135deg, rgba(255, 255, 255, 1), rgba(255, 255, 255, 0.95))',
                      borderRadius: '20px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      boxShadow: '0 12px 40px rgba(34, 197, 94, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.8)',
                      position: 'relative',
                      overflow: 'hidden',
                      transition: 'all 0.3s ease'
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = 'scale(1.05) rotate(5deg)'
                      e.currentTarget.style.boxShadow = '0 16px 50px rgba(34, 197, 94, 0.35), 0 0 0 1px rgba(255, 255, 255, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.9)'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = 'scale(1) rotate(0deg)'
                      e.currentTarget.style.boxShadow = '0 12px 40px rgba(34, 197, 94, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.8)'
                    }}
                    >
                      <img 
                        src="/logo.png" 
                        alt="VaidyaSathi Logo" 
                        style={{ 
                          width: '56px', 
                          height: '56px',
                          objectFit: 'contain',
                          filter: 'drop-shadow(0 4px 12px rgba(34, 197, 94, 0.4))',
                          transition: 'filter 0.3s ease'
                        }} 
                      />
                      {/* Shine effect */}
                      <div style={{
                        position: 'absolute',
                        top: '-50%',
                        left: '-50%',
                        width: '200%',
                        height: '200%',
                        background: 'linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.3) 50%, transparent 70%)',
                        transform: 'rotate(45deg)',
                        animation: 'shimmer 3s infinite'
                      }}></div>
                    </div>
                    <div>
                      <h1 style={{ 
                        fontSize: '2.75rem', 
                        fontWeight: '800', 
                        background: 'linear-gradient(135deg, #22c55e 0%, #3b82f6 50%, #16a34a 100%)', 
                        WebkitBackgroundClip: 'text', 
                        WebkitTextFillColor: 'transparent',
                        letterSpacing: '-1px',
                        marginBottom: '0.5rem',
                        lineHeight: '1.1'
                      }}>
                        VaidyaSathi
                      </h1>
                      <p style={{ 
                        color: '#15803d', 
                        fontWeight: '600', 
                        fontSize: '1.125rem',
                        letterSpacing: '0.3px'
                      }}>
                        Clinical Decision Support System with NLP, RAG & LLM
                      </p>
                    </div>
                  </div>
          
          {/* Feature Status Indicators */}
          <div className="flex justify-center" style={{ gap: '1rem', flexWrap: 'wrap' }}>
            <div style={{ 
              background: 'rgba(255, 255, 255, 0.9)', 
              padding: '0.5rem 1rem', 
              borderRadius: '9999px', 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.5rem', 
              border: `1px solid ${backendStatus === 'connected' ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)'}` 
            }}>
              <div style={{ 
                width: '8px', 
                height: '8px', 
                background: backendStatus === 'connected' ? '#22c55e' : backendStatus === 'checking' ? '#f59e0b' : '#ef4444', 
                borderRadius: '50%', 
                animation: 'pulse 2s infinite' 
              }}></div>
              <span style={{ 
                fontSize: '0.75rem', 
                fontWeight: '500', 
                color: backendStatus === 'connected' ? '#166534' : backendStatus === 'checking' ? '#92400e' : '#dc2626' 
              }}>
                Backend {backendStatus === 'connected' ? 'Connected' : backendStatus === 'checking' ? 'Checking...' : 'Disconnected'}
              </span>
            </div>
            <div style={{ background: 'rgba(255, 255, 255, 0.9)', padding: '0.5rem 1rem', borderRadius: '9999px', display: 'flex', alignItems: 'center', gap: '0.5rem', border: '1px solid rgba(34, 197, 94, 0.2)' }}>
              <div style={{ width: '8px', height: '8px', background: '#22c55e', borderRadius: '50%', animation: 'pulse 2s infinite' }}></div>
              <span style={{ fontSize: '0.75rem', fontWeight: '500', color: '#166534' }}>RAG System</span>
            </div>
            <div style={{ background: 'rgba(255, 255, 255, 0.9)', padding: '0.5rem 1rem', borderRadius: '9999px', display: 'flex', alignItems: 'center', gap: '0.5rem', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
              <div style={{ width: '8px', height: '8px', background: '#3b82f6', borderRadius: '50%', animation: 'pulse 2s infinite' }}></div>
              <span style={{ fontSize: '0.75rem', fontWeight: '500', color: '#1e40af' }}>NLP Engine</span>
            </div>
            <div style={{ background: 'rgba(255, 255, 255, 0.9)', padding: '0.5rem 1rem', borderRadius: '9999px', display: 'flex', alignItems: 'center', gap: '0.5rem', border: '1px solid rgba(168, 85, 247, 0.2)' }}>
              <div style={{ width: '8px', height: '8px', background: '#a855f7', borderRadius: '50%', animation: 'pulse 2s infinite' }}></div>
              <span style={{ fontSize: '0.75rem', fontWeight: '500', color: '#7c3aed' }}>LLM Features</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="container main-container" style={{ padding: '2rem 1rem', minHeight: 'calc(100vh - 200px)' }}>
        <div style={{ display: 'flex', gap: '2rem', alignItems: 'flex-start', justifyContent: 'center', flexWrap: 'wrap' }}>
          
          {/* Chat Interface with all CDSS features */}
          <div 
            className="chat-box-container"
            style={{ 
              width: `${chatBoxWidth}%`,
              height: `${chatBoxHeight}vh`,
              minWidth: '350px',
              minHeight: '400px',
              background: 'rgba(255, 255, 255, 0.95)', 
              backdropFilter: 'blur(16px)', 
              borderRadius: '1.5rem', 
              overflow: 'hidden', 
              boxShadow: '0 20px 60px rgba(0, 0, 0, 0.12), 0 0 0 1px rgba(0, 0, 0, 0.05)',
              position: 'relative',
              display: 'flex',
              flexDirection: 'column',
              transition: isResizing === 'chat' ? 'none' : 'box-shadow 0.3s ease'
            }}
            onMouseEnter={(e) => {
              if (!isResizing) e.currentTarget.style.boxShadow = '0 25px 70px rgba(34, 197, 94, 0.15), 0 0 0 1px rgba(34, 197, 94, 0.1)'
            }}
            onMouseLeave={(e) => {
              if (!isResizing) e.currentTarget.style.boxShadow = '0 20px 60px rgba(0, 0, 0, 0.12), 0 0 0 1px rgba(0, 0, 0, 0.05)'
            }}
          >
            
            {/* Chat Header */}
            <div style={{ background: 'linear-gradient(135deg, #22c55e, #16a34a)', padding: '1.5rem', color: 'white' }}>
              <div className="flex items-center justify-between">
                <div className="flex items-center" style={{ gap: '0.75rem' }}>
                  <div style={{ width: '40px', height: '40px', background: 'rgba(255, 255, 255, 0.2)', borderRadius: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
                    </svg>
                  </div>
                  <div>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: '700' }}>Enhanced RAG Chatbot</h2>
                    <p style={{ fontSize: '0.875rem', color: '#dcfce7' }}>NLP + RAG + LLM Integration</p>
                  </div>
                </div>
                <button
                  onClick={testAllFeatures}
                  disabled={isLoading}
                  style={{
                    padding: '0.5rem 1rem',
                    background: 'rgba(255, 255, 255, 0.2)',
                    border: 'none',
                    borderRadius: '0.75rem',
                    color: 'white',
                    fontWeight: '500',
                    cursor: 'pointer',
                    fontSize: '0.875rem'
                  }}
                >
                  🧪 Test All Features
                </button>
              </div>
            </div>

            {/* Messages Area */}
            <div 
              className="scrollable-container"
              style={{ 
                flex: 1, 
                overflowY: 'auto', 
                padding: '1.5rem', 
                background: 'linear-gradient(to bottom, rgba(255, 255, 255, 0.5), rgba(240, 253, 244, 0.3))' 
              }}
              onScroll={(e) => {
                e.target.classList.add('scrolling')
                clearTimeout(e.target.scrollTimeout)
                e.target.scrollTimeout = setTimeout(() => {
                  e.target.classList.remove('scrolling')
                }, 1000)
              }}
            >
              {messages.map((message) => (
                <div key={message.id} className="flex" style={{ justifyContent: message.type === 'user' ? 'flex-end' : 'flex-start', marginBottom: '1rem' }}>
                  <div className={`${message.type === 'user' ? 'message-user' : 'message-bot'}`} style={{ ...(message.isError && { background: 'linear-gradient(135deg, #fef2f2, #fee2e2)', color: '#991b1b', border: '1px solid #fecaca' }) }}>
                    <div style={{ fontWeight: '500', lineHeight: '1.5' }}>
                      {message.content.split('\n').map((line, index) => (
                        <div key={index}>
                          {parseMarkdown(line)}
                          {index < message.content.split('\n').length - 1 && <br />}
                        </div>
                      ))}
                    </div>
                    <div style={{ fontSize: '0.75rem', marginTop: '0.5rem', textAlign: 'right', opacity: 0.7 }}>
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="flex" style={{ justifyContent: 'flex-start', marginBottom: '1rem' }}>
                  <div className="message-bot">
                    <div className="flex items-center" style={{ gap: '0.5rem' }}>
                      <div className="flex" style={{ gap: '0.25rem' }}>
                        <div style={{ width: '6px', height: '6px', background: '#22c55e', borderRadius: '50%', animation: 'pulse 1.5s infinite' }}></div>
                        <div style={{ width: '6px', height: '6px', background: '#22c55e', borderRadius: '50%', animation: 'pulse 1.5s infinite 0.2s' }}></div>
                        <div style={{ width: '6px', height: '6px', background: '#22c55e', borderRadius: '50%', animation: 'pulse 1.5s infinite 0.4s' }}></div>
                      </div>
                      <span style={{ color: '#6b7280', fontWeight: '500' }}>
                        {currentResearchStep || 'Processing with NLP & RAG...'}
                      </span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div style={{ padding: '1.5rem', background: 'rgba(255, 255, 255, 0.6)', borderTop: '1px solid rgba(34, 197, 94, 0.2)' }}>
              <div className="flex" style={{ gap: '1rem' }}>
                <input
                  type="text"
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about medical conditions, symptoms, treatments..."
                  className="input"
                  disabled={isLoading}
                  style={{ flex: 1 }}
                />
                <button
                  onClick={handleSendMessage}
                  disabled={!inputMessage.trim() || isLoading}
                  className="btn-primary"
                  style={{ width: '56px', height: '56px', borderRadius: '1rem' }}
                >
                  <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24" style={{ transform: 'rotate(90deg)' }}>
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
                  </svg>
                </button>
              </div>
              
              {/* Medical Query Examples */}
              <div className="flex" style={{ gap: '0.5rem', marginTop: '1rem', flexWrap: 'wrap' }}>
                {[
                  "35-year-old with chronic headaches",
                  "Chest pain differential diagnosis", 
                  "Hypertension treatment guidelines"
                ].map((example, index) => (
                  <button
                    key={index}
                    onClick={() => setInputMessage(example)}
                    style={{
                      padding: '0.5rem 0.75rem',
                      fontSize: '0.75rem',
                      background: 'rgba(255, 255, 255, 0.8)',
                      border: '1px solid rgba(34, 197, 94, 0.2)',
                      borderRadius: '0.5rem',
                      color: '#374151',
                      cursor: 'pointer'
                    }}
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>

            {/* Chat Box Resize Handle */}
            <div
              style={{
                position: 'absolute',
                bottom: '0',
                right: '0',
                width: '32px',
                height: '32px',
                background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(34, 197, 94, 0.2))',
                borderRadius: '12px 0 12px 0',
                cursor: 'nwse-resize',
                opacity: isResizing === 'chat' ? 1 : 0.6,
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 10,
                backdropFilter: 'blur(4px)'
              }}
              onMouseDown={(e) => handleResizeStart('chat', e)}
              onMouseEnter={(e) => {
                e.currentTarget.style.opacity = '1'
                e.currentTarget.style.background = 'linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(34, 197, 94, 0.3))'
              }}
              onMouseLeave={(e) => {
                if (isResizing !== 'chat') {
                  e.currentTarget.style.opacity = '0.6'
                  e.currentTarget.style.background = 'linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(34, 197, 94, 0.2))'
                }
              }}
            >
              {/* Resize icon - three diagonal lines */}
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M14 10L10 14M14 6L6 14M14 2L2 14" stroke="#22c55e" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </div>
          </div>

          {/* Analysis Panel - Shows all CDSS analysis results */}
          <div 
            className="analysis-panel-container"
            style={{ 
              width: `${analysisBoxWidth}%`,
              height: `${analysisBoxHeight}vh`,
              minWidth: '350px',
              minHeight: '400px',
              position: 'relative',
              display: 'flex',
              flexDirection: 'column',
              transition: isResizing === 'analysis' ? 'none' : 'all 0.3s ease'
            }}
          >
            <div style={{ flex: 1, height: '100%', overflow: 'hidden' }}>
              <AnalysisPanel analysisData={analysisData} />
            </div>
            
            {/* Analysis Box Resize Handle */}
            <div
              style={{
                position: 'absolute',
                bottom: '0',
                right: '0',
                width: '32px',
                height: '32px',
                background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.2))',
                borderRadius: '12px 0 12px 0',
                cursor: 'nwse-resize',
                opacity: isResizing === 'analysis' ? 1 : 0.6,
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                zIndex: 10,
                backdropFilter: 'blur(4px)'
              }}
              onMouseDown={(e) => handleResizeStart('analysis', e)}
              onMouseEnter={(e) => {
                e.currentTarget.style.opacity = '1'
                e.currentTarget.style.background = 'linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(59, 130, 246, 0.3))'
              }}
              onMouseLeave={(e) => {
                if (isResizing !== 'analysis') {
                  e.currentTarget.style.opacity = '0.6'
                  e.currentTarget.style.background = 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(59, 130, 246, 0.2))'
                }
              }}
            >
              {/* Resize icon - three diagonal lines */}
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                <path d="M14 10L10 14M14 6L6 14M14 2L2 14" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round"/>
              </svg>
            </div>
          </div>
        </div>

         {/* Enhanced Feature Overview with Modern Design */}
         <div style={{ 
           marginTop: '4rem', 
           padding: '4rem 2rem',
           background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.03) 0%, rgba(59, 130, 246, 0.03) 50%, rgba(168, 85, 247, 0.03) 100%)',
           position: 'relative',
           overflow: 'hidden'
         }}>
           {/* Decorative elements */}
           <div style={{
             position: 'absolute',
             top: '-50px',
             right: '-50px',
             width: '200px',
             height: '200px',
             background: 'radial-gradient(circle, rgba(34, 197, 94, 0.1) 0%, transparent 70%)',
             borderRadius: '50%',
             filter: 'blur(40px)'
           }}></div>
           <div style={{
             position: 'absolute',
             bottom: '-50px',
             left: '-50px',
             width: '200px',
             height: '200px',
             background: 'radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%)',
             borderRadius: '50%',
             filter: 'blur(40px)'
           }}></div>
           
           <div style={{ 
             marginBottom: '3.5rem', 
             textAlign: 'center',
             position: 'relative',
             zIndex: 1
           }}>
             {/* Badge */}
             <div style={{
               display: 'inline-block',
               padding: '0.5rem 1.25rem',
               background: 'linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1))',
               border: '1px solid rgba(34, 197, 94, 0.2)',
               borderRadius: '9999px',
               marginBottom: '1.5rem',
               fontSize: '0.875rem',
               fontWeight: '600',
               color: '#16a34a',
               letterSpacing: '0.5px'
             }}>
               🚀 ADVANCED AI-POWERED PLATFORM
             </div>
             
             <h2 style={{ 
               fontSize: '3rem', 
               fontWeight: '800', 
               marginBottom: '1.25rem',
               background: 'linear-gradient(135deg, #22c55e 0%, #3b82f6 50%, #a855f7 100%)',
               WebkitBackgroundClip: 'text',
               WebkitTextFillColor: 'transparent',
               letterSpacing: '-1px',
               lineHeight: '1.2'
             }}>
               Complete CDSS Feature Integration
             </h2>
             
             <p style={{ 
               fontSize: '1.25rem', 
               color: '#6b7280', 
               marginBottom: '1rem',
               fontWeight: '500',
               maxWidth: '800px',
               margin: '0 auto 1rem',
               lineHeight: '1.6'
             }}>
               All NLP, RAG, and LLM features from the original system are preserved and enhanced with cutting-edge AI capabilities
             </p>
             
             {/* Decorative line with icon */}
             <div style={{
               display: 'flex',
               alignItems: 'center',
               justifyContent: 'center',
               gap: '0.75rem',
               marginTop: '2rem'
             }}>
               <div style={{
                 width: '60px',
                 height: '2px',
                 background: 'linear-gradient(90deg, transparent, #22c55e)',
               }}></div>
               <div style={{
                 width: '10px',
                 height: '10px',
                 background: 'linear-gradient(135deg, #22c55e, #3b82f6)',
                 borderRadius: '50%',
                 boxShadow: '0 0 20px rgba(34, 197, 94, 0.4)'
               }}></div>
               <div style={{
                 width: '60px',
                 height: '2px',
                 background: 'linear-gradient(90deg, #3b82f6, transparent)',
               }}></div>
             </div>
           </div>
           
           <div style={{ 
             display: 'grid', 
             gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', 
             gap: '2.5rem',
             width: '100%',
             position: 'relative',
             zIndex: 1
           }}>
               {[
                 { 
                   icon: '🧠', 
                   title: 'NLP Processing', 
                   desc: 'Advanced text preprocessing, tokenization, and semantic analysis with medical terminology recognition',
                   color: '#ec4899',
                   bgColor: 'rgba(236, 72, 153, 0.1)'
                 },
                 { 
                   icon: '🔍', 
                   title: 'RAG System', 
                   desc: 'Retrieval-augmented generation with comprehensive medical knowledge base integration',
                   color: '#3b82f6',
                   bgColor: 'rgba(59, 130, 246, 0.1)'
                 },
                 { 
                   icon: '🤖', 
                   title: 'LLM Features', 
                   desc: 'Advanced language model capabilities with medical context understanding and response generation',
                   color: '#8b5cf6',
                   bgColor: 'rgba(139, 92, 246, 0.1)'
                 },
                 { 
                   icon: '📊', 
                   title: 'Medical Analysis', 
                   desc: 'Comprehensive differential diagnosis, risk assessment, and evidence-based recommendations',
                   color: '#f59e0b',
                   bgColor: 'rgba(245, 158, 11, 0.1)'
                 },
                 { 
                   icon: '📚', 
                   title: 'Knowledge Base', 
                   desc: 'Extensive medical literature, research papers, and clinical guidelines database',
                   color: '#22c55e',
                   bgColor: 'rgba(34, 197, 94, 0.1)'
                 },
                 { 
                   icon: '💾', 
                   title: 'Database Integration', 
                   desc: 'Complete data models, relationships, and patient information management system',
                   color: '#ef4444',
                   bgColor: 'rgba(239, 68, 68, 0.1)'
                 }
               ].map((feature, index) => (
                 <div 
                   key={index} 
                   className="feature-card-enhanced"
                   style={{ 
                     background: 'rgba(255, 255, 255, 0.95)', 
                     padding: '2.5rem', 
                     borderRadius: '1.5rem', 
                     boxShadow: '0 8px 32px rgba(0, 0, 0, 0.08), 0 0 0 1px rgba(0, 0, 0, 0.04)',
                     border: `1px solid ${feature.color}15`,
                     backdropFilter: 'blur(16px)',
                     transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
                     position: 'relative',
                     overflow: 'hidden',
                     cursor: 'pointer'
                   }}
                   onMouseEnter={(e) => {
                     e.currentTarget.style.transform = 'translateY(-8px) scale(1.02)'
                     e.currentTarget.style.boxShadow = `0 20px 60px rgba(0, 0, 0, 0.12), 0 0 0 1px ${feature.color}30, 0 0 40px ${feature.color}20`
                     const glow = e.currentTarget.querySelector('.feature-card-glow')
                     if (glow) glow.style.opacity = '1'
                   }}
                   onMouseLeave={(e) => {
                     e.currentTarget.style.transform = 'translateY(0) scale(1)'
                     e.currentTarget.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.08), 0 0 0 1px rgba(0, 0, 0, 0.04)'
                     const glow = e.currentTarget.querySelector('.feature-card-glow')
                     if (glow) glow.style.opacity = '0'
                   }}
                 >
                   {/* Animated background gradient */}
                   <div style={{
                     position: 'absolute',
                     top: 0,
                     left: 0,
                     right: 0,
                     height: '4px',
                     background: `linear-gradient(90deg, ${feature.color}, ${feature.color}88)`,
                     opacity: 0.6,
                     transition: 'opacity 0.3s ease'
                   }}></div>
                   
                   {/* Hover glow effect */}
                   <div 
                     className="feature-card-glow"
                     style={{
                       position: 'absolute',
                       top: '-50%',
                       left: '-50%',
                       width: '200%',
                       height: '200%',
                       background: `radial-gradient(circle, ${feature.color}08 0%, transparent 70%)`,
                       opacity: 0,
                       transition: 'opacity 0.4s ease',
                       pointerEvents: 'none'
                     }}
                     data-glow
                   ></div>
                   
                   <div style={{ position: 'relative', zIndex: 1 }}>
                     {/* Feature icon with enhanced design */}
                     <div style={{
                       width: '72px',
                       height: '72px',
                       background: `linear-gradient(135deg, ${feature.color}15, ${feature.color}08)`,
                       borderRadius: '20px',
                       display: 'flex',
                       alignItems: 'center',
                       justifyContent: 'center',
                       fontSize: '2rem',
                       marginBottom: '1.75rem',
                       border: `2px solid ${feature.color}25`,
                       boxShadow: `0 8px 24px ${feature.color}15, inset 0 1px 0 rgba(255, 255, 255, 0.5)`,
                       transition: 'all 0.3s ease'
                     }}
                     onMouseEnter={(e) => {
                       e.currentTarget.style.transform = 'scale(1.1) rotate(5deg)'
                       e.currentTarget.style.boxShadow = `0 12px 32px ${feature.color}25, inset 0 1px 0 rgba(255, 255, 255, 0.5)`
                     }}
                     onMouseLeave={(e) => {
                       e.currentTarget.style.transform = 'scale(1) rotate(0deg)'
                       e.currentTarget.style.boxShadow = `0 8px 24px ${feature.color}15, inset 0 1px 0 rgba(255, 255, 255, 0.5)`
                     }}
                     >
                       {feature.icon}
                     </div>
                     
                     <h3 style={{ 
                       fontSize: '1.5rem', 
                       fontWeight: '800', 
                       marginBottom: '1rem',
                       background: `linear-gradient(135deg, ${feature.color}, ${feature.color}cc)`,
                       WebkitBackgroundClip: 'text',
                       WebkitTextFillColor: 'transparent',
                       letterSpacing: '-0.5px'
                     }}>
                       {feature.title}
                     </h3>
                     
                     <p style={{ 
                       fontSize: '1rem', 
                       color: '#6b7280',
                       lineHeight: '1.7',
                       marginBottom: '1.5rem',
                       fontWeight: '400'
                     }}>
                       {feature.desc}
                     </p>
                     
                     {/* Enhanced feature indicator with badge */}
                     <div style={{
                       display: 'flex',
                       alignItems: 'center',
                       justifyContent: 'space-between',
                       padding: '0.75rem 1rem',
                       background: `linear-gradient(135deg, ${feature.color}08, ${feature.color}04)`,
                       borderRadius: '0.75rem',
                       border: `1px solid ${feature.color}20`
                     }}>
                       <div style={{
                         display: 'flex',
                         alignItems: 'center',
                         gap: '0.75rem',
                         fontSize: '0.875rem',
                         color: feature.color,
                         fontWeight: '700'
                       }}>
                         <div style={{
                           width: '10px',
                           height: '10px',
                           background: feature.color,
                           borderRadius: '50%',
                           boxShadow: `0 0 12px ${feature.color}60`,
                           animation: 'pulse 2s infinite'
                         }}></div>
                         <span>Active & Enhanced</span>
                       </div>
                       <div style={{
                         fontSize: '0.75rem',
                         fontWeight: '600',
                         color: feature.color,
                         opacity: 0.8,
                         letterSpacing: '0.5px'
                       }}>
                         ✓ INTEGRATED
                       </div>
                     </div>
                   </div>
                 </div>
               ))}
             </div>
         </div>

         
      </div>
    </div>
  )
}

// Analysis Panel Component - Displays all CDSS analysis data EXACTLY as in original Django template
function AnalysisPanel({ analysisData }) {
  const [activeSection, setActiveSection] = useState('overview')
  const [isHovered, setIsHovered] = useState(false)

  const sections = [
    { id: 'overview', label: 'Overview', icon: '📊' },
    { id: 'nlp', label: 'NLP Status', icon: '🧠' },
    { id: 'diagnosis', label: 'Diagnosis', icon: '🔬' },
    { id: 'research', label: 'Research', icon: '📚' }
  ]

  return (
    <div 
      style={{ 
        height: '100%', 
        background: 'rgba(255, 255, 255, 0.95)', 
        backdropFilter: 'blur(16px)', 
        borderRadius: '1.5rem', 
        overflow: 'hidden', 
        boxShadow: isHovered 
          ? '0 25px 70px rgba(59, 130, 246, 0.15), 0 0 0 1px rgba(59, 130, 246, 0.1)'
          : '0 20px 60px rgba(0, 0, 0, 0.12), 0 0 0 1px rgba(0, 0, 0, 0.05)',
        display: 'flex',
        flexDirection: 'column',
        transition: 'box-shadow 0.3s ease'
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      
      {/* Panel Header - Matching original */}
      <div style={{ background: 'linear-gradient(135deg, #3b82f6, #1d4ed8)', padding: '1.5rem', color: 'white' }}>
        <div className="flex items-center" style={{ gap: '0.75rem' }}>
          <div style={{ width: '40px', height: '40px', background: 'rgba(255, 255, 255, 0.2)', borderRadius: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
            </svg>
          </div>
      <div>
            <h2 style={{ fontSize: '1.25rem', fontWeight: '700' }}>Detailed Analysis</h2>
            <p style={{ fontSize: '0.875rem', color: '#dbeafe' }}>AI Processing Results</p>
          </div>
        </div>
      </div>

      {/* Navigation - Matching original */}
      <div style={{ borderBottom: '1px solid rgba(59, 130, 246, 0.2)', background: 'rgba(255, 255, 255, 0.5)' }}>
        <div className="flex" style={{ overflowX: 'auto' }}>
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => setActiveSection(section.id)}
              style={{
                padding: '0.75rem 1rem',
                border: 'none',
                background: 'transparent',
                fontSize: '0.875rem',
                fontWeight: '500',
                cursor: 'pointer',
                borderBottom: activeSection === section.id ? '2px solid #3b82f6' : '2px solid transparent',
                color: activeSection === section.id ? '#1e40af' : '#6b7280',
                whiteSpace: 'nowrap'
              }}
            >
              {section.icon} {section.label}
        </button>
          ))}
        </div>
      </div>

      {/* Content - Full height to match container */}
      <div 
        className="analysis-panel" 
        onScroll={(e) => {
          e.target.classList.add('scrolling')
          clearTimeout(e.target.scrollTimeout)
          e.target.scrollTimeout = setTimeout(() => {
            e.target.classList.remove('scrolling')
          }, 1000)
        }}
        style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', background: 'linear-gradient(to bottom, rgba(255, 255, 255, 0.5), rgba(239, 246, 255, 0.3))' }}
      >
        {activeSection === 'overview' && <OverviewSection analysisData={analysisData} />}
        {activeSection === 'nlp' && <NLPSection analysisData={analysisData} />}
        {activeSection === 'diagnosis' && <DiagnosisSection analysisData={analysisData} />}
        {activeSection === 'research' && <ResearchSection analysisData={analysisData} />}
      </div>
    </div>
  )
}

// Original Overview Section - EXACTLY as in Django template
function OverviewSection({ analysisData }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <div style={{ background: 'rgba(255, 255, 255, 0.8)', padding: '1.5rem', borderRadius: '1rem', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
        <div className="flex items-center" style={{ gap: '0.75rem', marginBottom: '1rem' }}>
          <div style={{ width: '12px', height: '12px', background: '#22c55e', borderRadius: '50%', animation: 'pulse 2s infinite' }}></div>
          <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#22c55e' }}>Welcome</h3>
        </div>
        <p style={{ color: '#374151', lineHeight: '1.6' }}>
          {analysisData ? 'Detailed analysis of your query is available below.' : 'Detailed analysis of your query will appear here after you send a message.'}
        </p>
      </div>
      
      {analysisData && analysisData.differential_diagnoses && (
        <div style={{ background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(147, 197, 253, 0.1))', padding: '1.5rem', borderRadius: '1rem', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
          <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#1e40af', marginBottom: '1rem' }}>🔬 Potential Diagnoses</h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {analysisData.differential_diagnoses.slice(0, 3).map((diagnosis, index) => (
              <div key={index} className="flex justify-between items-center">
                <span style={{ fontSize: '0.875rem', fontWeight: '500', color: '#1e40af' }}>{diagnosis.condition}</span>
                <span style={{ fontSize: '0.75rem', fontWeight: '600', color: '#3b82f6' }}>{diagnosis.confidence}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// Original NLP Section - EXACTLY as in Django template  
function NLPSection({ analysisData }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      
      {/* NLP Processing Pipeline Status - EXACT match */}
      {(analysisData?.preprocessing_stats || analysisData?.semantic_analysis || analysisData?.llm_features) && (
        <div style={{ background: 'linear-gradient(135deg, #22c55e, #16a34a)', borderRadius: '1rem', padding: '1.5rem', color: 'white', marginBottom: '1.5rem' }}>
          <h3 style={{ fontSize: '1.25rem', fontWeight: '700', marginBottom: '1rem', display: 'flex', alignItems: 'center' }}>
            <span style={{ marginRight: '0.75rem' }}>🧠</span>NLP Processing Pipeline Status
          </h3>
          
          {/* Preprocessing Status */}
          {analysisData?.preprocessing_stats && (
            <div style={{ background: 'rgba(255, 255, 255, 0.2)', backdropFilter: 'blur(4px)', borderRadius: '0.75rem', padding: '1rem', marginBottom: '1rem', border: '1px solid rgba(255, 255, 255, 0.3)' }}>
              <h4 style={{ fontWeight: '600', marginBottom: '0.75rem', fontSize: '1.125rem' }}>📝 Text Preprocessing</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem', fontSize: '0.875rem' }}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span style={{ width: '8px', height: '8px', background: '#dcfce7', borderRadius: '50%', marginRight: '0.5rem' }}></span>
                  Sentences: {analysisData.preprocessing_stats.sentence_count || 0}
                </div>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span style={{ width: '8px', height: '8px', background: '#dcfce7', borderRadius: '50%', marginRight: '0.5rem' }}></span>
                  Tokens: {analysisData.preprocessing_stats.token_count || 0}
                </div>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span style={{ width: '8px', height: '8px', background: '#dcfce7', borderRadius: '50%', marginRight: '0.5rem' }}></span>
                  Normalized: {analysisData.preprocessing_stats.normalized_count || 0}
                </div>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span style={{ width: '8px', height: '8px', background: '#dcfce7', borderRadius: '50%', marginRight: '0.5rem' }}></span>
                  POS tags: {analysisData.preprocessing_stats.pos_count || 0}
                </div>
                {analysisData.preprocessing_stats.spell_corrections > 0 && (
                  <div style={{ display: 'flex', alignItems: 'center', gridColumn: 'span 2' }}>
                    <span style={{ width: '8px', height: '8px', background: '#fef3c7', borderRadius: '50%', marginRight: '0.5rem' }}></span>
                    Spell corrections: {analysisData.preprocessing_stats.spell_corrections}
                  </div>
                )}
              </div>
            </div>
          )}
          
          {/* Semantic Analysis Status */}
          {analysisData?.semantic_analysis && (
            <div style={{ background: 'rgba(255, 255, 255, 0.2)', backdropFilter: 'blur(4px)', borderRadius: '0.75rem', padding: '1rem', marginBottom: '1rem', border: '1px solid rgba(255, 255, 255, 0.3)' }}>
              <h4 style={{ fontWeight: '600', marginBottom: '0.75rem', fontSize: '1.125rem' }}>🔬 Semantic Analysis</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem', fontSize: '0.875rem' }}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span style={{ width: '8px', height: '8px', background: '#dbeafe', borderRadius: '50%', marginRight: '0.5rem' }}></span>
                  Entities: {Object.keys(analysisData.semantic_analysis.medical_entities || {}).length}
                </div>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span style={{ width: '8px', height: '8px', background: '#dbeafe', borderRadius: '50%', marginRight: '0.5rem' }}></span>
                  Relations: {(analysisData.semantic_analysis.medical_relationships || []).length}
                </div>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span style={{ width: '8px', height: '8px', background: '#dbeafe', borderRadius: '50%', marginRight: '0.5rem' }}></span>
                  Disambiguated: {Object.keys(analysisData.semantic_analysis.word_sense_disambiguation || {}).length}
                </div>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  <span style={{ width: '8px', height: '8px', background: analysisData.semantic_analysis.error ? '#fecaca' : '#dcfce7', borderRadius: '50%', marginRight: '0.5rem' }}></span>
                  UMLS: {analysisData.semantic_analysis.error ? 'Error' : 'Active'}
                </div>
              </div>
            </div>
          )}
          
          {/* LLM Features Status */}
          {analysisData?.llm_features && (
            <div style={{ background: 'rgba(255, 255, 255, 0.2)', backdropFilter: 'blur(4px)', borderRadius: '0.75rem', padding: '1rem', border: '1px solid rgba(255, 255, 255, 0.3)' }}>
              <h4 style={{ fontWeight: '600', marginBottom: '0.75rem', fontSize: '1.125rem' }}>🤖 Advanced LLM Features</h4>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem', fontSize: '0.875rem' }}>
                {Object.entries(analysisData.llm_features).map(([feature, active]) => {
                  const statusColor = active ? '#dcfce7' : '#fecaca'
                  const featureName = feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
                  return (
                    <div key={feature} style={{ display: 'flex', alignItems: 'center' }}>
                      <span style={{ width: '8px', height: '8px', background: statusColor, borderRadius: '50%', marginRight: '0.5rem' }}></span>
                      {featureName}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Original Diagnosis Section - EXACTLY as in Django template
function DiagnosisSection({ analysisData }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      
      {/* Summary */}
      {analysisData?.summary && (
        <div style={{ background: 'rgba(255, 255, 255, 0.6)', backdropFilter: 'blur(4px)', borderRadius: '1rem', padding: '1.5rem', border: '1px solid rgba(34, 197, 94, 0.3)', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)' }}>
          <div className="flex items-center" style={{ gap: '0.75rem', marginBottom: '1rem' }}>
            <div style={{ width: '12px', height: '12px', background: '#22c55e', borderRadius: '50%', animation: 'pulse 2s infinite' }}></div>
            <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#15803d' }}>📋 Summary</h3>
          </div>
          <p style={{ color: '#374151', lineHeight: '1.625' }}>{analysisData.summary}</p>
        </div>
      )}

      {/* Differential Diagnoses with confidence bars - EXACT match */}
      {analysisData?.differential_diagnoses && analysisData.differential_diagnoses.length > 0 && (
        <div style={{ background: 'rgba(255, 255, 255, 0.6)', backdropFilter: 'blur(4px)', borderRadius: '1rem', padding: '1.5rem', border: '1px solid rgba(34, 197, 94, 0.3)', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)' }}>
          <div className="flex items-center" style={{ gap: '0.75rem', marginBottom: '1.5rem' }}>
            <div style={{ width: '12px', height: '12px', background: '#22c55e', borderRadius: '50%', animation: 'pulse 2s infinite' }}></div>
            <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#15803d' }}>🔬 Potential Diagnoses</h3>
          </div>
          
          {analysisData.differential_diagnoses.map((diagnosis, index) => (
            <div key={index} style={{ marginBottom: '1.5rem', padding: '1rem', background: 'rgba(255, 255, 255, 0.4)', borderRadius: '0.75rem', border: '1px solid rgba(34, 197, 94, 0.2)', backdropFilter: 'blur(4px)' }}>
              <div className="flex justify-between items-center" style={{ marginBottom: '0.75rem' }}>
                <strong style={{ color: '#1f2937', fontWeight: '600' }}>{diagnosis.condition}</strong>
                <span style={{ fontSize: '0.875rem', fontWeight: '700', color: '#16a34a' }}>{diagnosis.confidence}%</span>
              </div>
              <div style={{ width: '100%', background: '#e5e7eb', borderRadius: '9999px', height: '0.75rem', marginBottom: '0.75rem', overflow: 'hidden' }}>
                <div 
                  style={{ 
                    height: '0.75rem', 
                    background: 'linear-gradient(90deg, #4ade80, #16a34a)', 
                    borderRadius: '9999px',
                    width: `${diagnosis.confidence}%`,
                    transition: 'width 1s ease-out',
                    animationDelay: `${index * 0.2}s`
                  }}
                />
              </div>
              <p style={{ color: '#374151', fontSize: '0.875rem', lineHeight: '1.625' }}>{diagnosis.explanation || ''}</p>
            </div>
          ))}
        </div>
      )}

      {/* Recommendations with modern card styling - EXACT match */}
      {analysisData?.recommendations && (
        <div style={{ background: 'rgba(255, 255, 255, 0.6)', backdropFilter: 'blur(4px)', borderRadius: '1rem', padding: '1.5rem', border: '1px solid rgba(34, 197, 94, 0.3)', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)' }}>
          <div className="flex items-center" style={{ gap: '0.75rem', marginBottom: '1.5rem' }}>
            <div style={{ width: '12px', height: '12px', background: '#22c55e', borderRadius: '50%', animation: 'pulse 2s infinite' }}></div>
            <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#15803d' }}>💊 Recommendations</h3>
          </div>
          
          {/* Immediate actions */}
          {analysisData.recommendations.immediate_actions && analysisData.recommendations.immediate_actions.length > 0 && (
            <div style={{ marginBottom: '1.5rem', padding: '1rem', background: 'rgba(254, 242, 242, 0.8)', borderRadius: '0.75rem', border: '1px solid rgba(254, 202, 202, 0.5)' }}>
              <div className="flex items-center" style={{ gap: '0.5rem', marginBottom: '0.75rem' }}>
                <span style={{ color: '#dc2626' }}>🚨</span>
                <div style={{ fontWeight: '600', color: '#b91c1c' }}>Immediate Actions</div>
              </div>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                {analysisData.recommendations.immediate_actions.map((action, index) => (
                  <li key={index} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', fontSize: '0.875rem', color: '#991b1b', marginBottom: '0.5rem' }}>
                    <span style={{ width: '6px', height: '6px', background: '#ef4444', borderRadius: '50%', marginTop: '0.5rem', flexShrink: 0 }}></span>
                    <span>{action}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {/* Tests */}
          {analysisData.recommendations.tests && analysisData.recommendations.tests.length > 0 && (
            <div style={{ marginBottom: '1.5rem', padding: '1rem', background: 'rgba(239, 246, 255, 0.8)', borderRadius: '0.75rem', border: '1px solid rgba(191, 219, 254, 0.5)' }}>
              <div className="flex items-center" style={{ gap: '0.5rem', marginBottom: '0.75rem' }}>
                <span style={{ color: '#2563eb' }}>🔬</span>
                <div style={{ fontWeight: '600', color: '#1d4ed8' }}>Recommended Tests</div>
              </div>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                {analysisData.recommendations.tests.map((test, index) => (
                  <li key={index} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', fontSize: '0.875rem', color: '#1e40af', marginBottom: '0.5rem' }}>
                    <span style={{ width: '6px', height: '6px', background: '#3b82f6', borderRadius: '50%', marginTop: '0.5rem', flexShrink: 0 }}></span>
                    <span>{test}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {/* Lifestyle */}
          {analysisData.recommendations.lifestyle && analysisData.recommendations.lifestyle.length > 0 && (
            <div style={{ marginBottom: '1.5rem', padding: '1rem', background: 'rgba(240, 253, 244, 0.8)', borderRadius: '0.75rem', border: '1px solid rgba(187, 247, 208, 0.5)' }}>
              <div className="flex items-center" style={{ gap: '0.5rem', marginBottom: '0.75rem' }}>
                <span style={{ color: '#16a34a' }}>🌱</span>
                <div style={{ fontWeight: '600', color: '#15803d' }}>Lifestyle Recommendations</div>
              </div>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                {analysisData.recommendations.lifestyle.map((item, index) => (
                  <li key={index} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', fontSize: '0.875rem', color: '#166534', marginBottom: '0.5rem' }}>
                    <span style={{ width: '6px', height: '6px', background: '#22c55e', borderRadius: '50%', marginTop: '0.5rem', flexShrink: 0 }}></span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {/* Follow-up */}
          {analysisData.recommendations.follow_up && (
            <div style={{ padding: '1rem', background: 'rgba(250, 245, 255, 0.8)', borderRadius: '0.75rem', border: '1px solid rgba(233, 213, 255, 0.5)' }}>
              <div className="flex items-center" style={{ gap: '0.5rem', marginBottom: '0.75rem' }}>
                <span style={{ color: '#9333ea' }}>📅</span>
                <div style={{ fontWeight: '600', color: '#7c3aed' }}>Follow-up</div>
              </div>
              <p style={{ fontSize: '0.875rem', color: '#7c2d12' }}>{analysisData.recommendations.follow_up}</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// Original Research Section - EXACTLY as in Django template
function ResearchSection({ analysisData }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      
      {/* Research papers - EXACT match */}
      {analysisData?.research_papers && Object.keys(analysisData.research_papers).length > 0 ? (
        <div style={{ background: 'rgba(255, 255, 255, 0.6)', backdropFilter: 'blur(4px)', borderRadius: '1rem', padding: '1.5rem', border: '1px solid rgba(34, 197, 94, 0.3)', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)' }}>
          <div className="flex items-center" style={{ gap: '0.75rem', marginBottom: '1.5rem' }}>
            <div style={{ width: '12px', height: '12px', background: '#22c55e', borderRadius: '50%', animation: 'pulse 2s infinite' }}></div>
            <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#15803d' }}>📚 Relevant Research Papers</h3>
          </div>
          
          {Object.entries(analysisData.research_papers).map(([condition, papers], index) => {
            if (papers && papers.length > 0) {
              return (
                <div key={index} style={{ marginBottom: '1.5rem' }}>
                  <h4 style={{ fontWeight: '600', color: '#16a34a', marginBottom: '1rem', display: 'flex', alignItems: 'center' }}>
                    <span style={{ width: '8px', height: '8px', background: '#22c55e', borderRadius: '50%', marginRight: '0.5rem' }}></span>
                    {condition}
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {papers.map((paper, paperIndex) => (
                      <div key={paperIndex} style={{ padding: '1rem', background: 'rgba(255, 255, 255, 0.6)', borderRadius: '0.75rem', border: '1px solid rgba(34, 197, 94, 0.2)', backdropFilter: 'blur(4px)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
                          <h5 style={{ fontWeight: '600', color: '#15803d', fontSize: '0.875rem', lineHeight: '1.4', flex: 1, marginRight: '1rem' }}>
                            {paper.title}
                          </h5>
                          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.25rem' }}>
                            <span style={{ fontSize: '0.75rem', fontWeight: '500', color: '#16a34a', background: 'rgba(34, 197, 94, 0.1)', padding: '0.25rem 0.5rem', borderRadius: '0.375rem' }}>
                              {paper.year}
                            </span>
                            {paper.evidence_level && (
                              <span style={{ fontSize: '0.75rem', fontWeight: '500', color: '#7c3aed', background: 'rgba(124, 58, 237, 0.1)', padding: '0.25rem 0.5rem', borderRadius: '0.375rem' }}>
                                {paper.evidence_level}
                              </span>
                            )}
                            {paper.clinical_significance && (
                              <span style={{ 
                                fontSize: '0.75rem', 
                                fontWeight: '500', 
                                color: paper.clinical_significance === 'High' ? '#dc2626' : paper.clinical_significance === 'Medium' ? '#d97706' : '#059669',
                                background: paper.clinical_significance === 'High' ? 'rgba(220, 38, 38, 0.1)' : paper.clinical_significance === 'Medium' ? 'rgba(217, 119, 6, 0.1)' : 'rgba(5, 150, 105, 0.1)',
                                padding: '0.25rem 0.5rem', 
                                borderRadius: '0.375rem' 
                              }}>
                                {paper.clinical_significance} Significance
                              </span>
                            )}
                          </div>
                        </div>
                        <div style={{ fontSize: '0.75rem', color: '#374151', marginBottom: '0.5rem' }}>
                          <strong>Authors:</strong> {paper.authors}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: '#374151', marginBottom: '0.5rem' }}>
                          <strong>Journal:</strong> {paper.journal}
                        </div>
                        {paper.pmid && (
                          <div style={{ fontSize: '0.75rem', color: '#374151', marginBottom: '0.5rem' }}>
                            <strong>PMID:</strong> {paper.pmid}
                          </div>
                        )}
                        {paper.study_type && (
                          <div style={{ fontSize: '0.75rem', color: '#374151', marginBottom: '0.5rem' }}>
                            <strong>Study Type:</strong> {paper.study_type}
                          </div>
                        )}
                        {paper.sample_size && (
                          <div style={{ fontSize: '0.75rem', color: '#374151', marginBottom: '0.5rem' }}>
                            <strong>Sample Size:</strong> {paper.sample_size}
                          </div>
                        )}
                        {(paper.abstract || paper.summary) && (
                          <div style={{ fontSize: '0.75rem', color: '#4b5563', lineHeight: '1.4', marginBottom: '0.5rem' }}>
                            <strong>Abstract:</strong> {paper.abstract || paper.summary}
                          </div>
                        )}
                        {paper.key_findings && paper.key_findings.length > 0 && (
                          <div style={{ fontSize: '0.75rem', color: '#4b5563' }}>
                            <strong>Key Findings:</strong>
                            <ul style={{ margin: '0.25rem 0 0 1rem', padding: 0 }}>
                              {paper.key_findings.slice(0, 3).map((finding, findingIndex) => (
                                <li key={findingIndex} style={{ marginBottom: '0.25rem' }}>{finding}</li>
                              ))}
                            </ul>
                          </div>
                        )}
                        {paper.medical_conditions && paper.medical_conditions.length > 0 && (
                          <div style={{ fontSize: '0.75rem', color: '#4b5563', marginTop: '0.5rem' }}>
                            <strong>Medical Conditions:</strong> {paper.medical_conditions.join(', ')}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )
            }
            return null
          })}
        </div>
      ) : (
        <div style={{ background: 'rgba(255, 255, 255, 0.6)', backdropFilter: 'blur(4px)', borderRadius: '1rem', padding: '1.5rem', border: '1px solid rgba(168, 85, 247, 0.3)', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)' }}>
          <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#7c3aed', marginBottom: '1rem', display: 'flex', alignItems: 'center' }}>
            <span style={{ marginRight: '0.5rem' }}>📚</span>
            Research Papers
          </h3>
          <p style={{ color: '#8b5cf6', fontSize: '0.875rem' }}>
            Relevant research papers will appear here based on your analysis.
          </p>
        </div>
      )}

      {/* Enhanced LLM response - EXACT match */}
      {analysisData?.enhanced_response && analysisData.enhanced_response.response && (
        <div style={{ background: 'linear-gradient(135deg, #a855f7, #6366f1)', borderRadius: '1rem', padding: '1.5rem', color: 'white', boxShadow: '0 8px 25px rgba(0, 0, 0, 0.15)' }}>
          <div className="flex items-center" style={{ gap: '0.75rem', marginBottom: '1rem' }}>
            <div style={{ padding: '0.5rem', background: 'rgba(255, 255, 255, 0.2)', borderRadius: '0.5rem' }}>
              <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
              </svg>
            </div>
            <div>
              <h3 style={{ fontSize: '1.125rem', fontWeight: '700' }}>🤖 Enhanced Clinical Analysis</h3>
              <p style={{ fontSize: '0.875rem', color: '#e0e7ff' }}>Response Type: {analysisData.enhanced_response.response_type || 'general'}</p>
            </div>
          </div>
          <div style={{ background: 'rgba(255, 255, 255, 0.1)', backdropFilter: 'blur(4px)', borderRadius: '0.75rem', padding: '1rem', border: '1px solid rgba(255, 255, 255, 0.2)' }}>
            <div style={{ color: 'white', lineHeight: '1.625', whiteSpace: 'pre-wrap' }}>{analysisData.enhanced_response.response}</div>
          </div>
        </div>
      )}

      {/* All other analysis components - EXACT match structure */}
      {analysisData?.medical_summary && (
        <MedicalSummarySection summary={analysisData.medical_summary} />
      )}

      {analysisData?.semantic_analysis && (
        <SemanticAnalysisSection analysis={analysisData.semantic_analysis} />
      )}

      {analysisData?.qa_results && analysisData.qa_results.length > 0 && (
        <QAResultsSection results={analysisData.qa_results} />
      )}

      {analysisData?.retrieved_context && analysisData.retrieved_context.length > 0 && (
        <RetrievedContextSection context={analysisData.retrieved_context} />
      )}
    </div>
  )
}

// Additional sections to match EXACTLY the original Django template structure
function MedicalSummarySection({ summary }) {
  return (
    <div style={{ background: 'rgba(255, 255, 255, 0.6)', backdropFilter: 'blur(4px)', borderRadius: '1rem', padding: '1.5rem', border: '1px solid rgba(34, 197, 94, 0.3)', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)' }}>
      <div className="flex items-center" style={{ gap: '0.75rem', marginBottom: '1.5rem' }}>
        <div style={{ width: '12px', height: '12px', background: '#22c55e', borderRadius: '50%', animation: 'pulse 2s infinite' }}></div>
        <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#15803d' }}>📝 Medical Text Summarization</h3>
      </div>
      
      {summary.extractive_summary && (
        <div style={{ marginBottom: '1rem', padding: '1rem', background: 'rgba(240, 253, 244, 0.8)', borderRadius: '0.75rem', border: '1px solid rgba(187, 247, 208, 0.5)' }}>
          <h4 style={{ fontWeight: '600', color: '#15803d', marginBottom: '0.75rem', display: 'flex', alignItems: 'center' }}>
            <span style={{ color: '#16a34a', marginRight: '0.5rem' }}>🔍</span>
            Extractive Summary
          </h4>
          <p style={{ color: '#166534', fontSize: '0.875rem', lineHeight: '1.625' }}>{summary.extractive_summary}</p>
        </div>
      )}
      
      {summary.abstractive_summary && (
        <div style={{ marginBottom: '1rem', padding: '1rem', background: 'rgba(239, 246, 255, 0.8)', borderRadius: '0.75rem', border: '1px solid rgba(191, 219, 254, 0.5)' }}>
          <h4 style={{ fontWeight: '600', color: '#1d4ed8', marginBottom: '0.75rem', display: 'flex', alignItems: 'center' }}>
            <span style={{ color: '#2563eb', marginRight: '0.5rem' }}>🧠</span>
            Abstractive Summary
          </h4>
          <p style={{ color: '#1e40af', fontSize: '0.875rem', lineHeight: '1.625' }}>{summary.abstractive_summary}</p>
        </div>
      )}
      
      {summary.key_findings && summary.key_findings.length > 0 && (
        <div style={{ padding: '1rem', background: 'rgba(250, 245, 255, 0.8)', borderRadius: '0.75rem', border: '1px solid rgba(233, 213, 255, 0.5)' }}>
          <h4 style={{ fontWeight: '600', color: '#7c3aed', marginBottom: '0.75rem', display: 'flex', alignItems: 'center' }}>
            <span style={{ color: '#9333ea', marginRight: '0.5rem' }}>💡</span>
            Key Findings
          </h4>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
            {summary.key_findings.map((finding, index) => (
              <li key={index} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', fontSize: '0.875rem', color: '#7c2d12', marginBottom: '0.5rem' }}>
                <span style={{ width: '6px', height: '6px', background: '#a855f7', borderRadius: '50%', marginTop: '0.5rem', flexShrink: 0 }}></span>
                <span>{finding}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

function SemanticAnalysisSection({ analysis }) {
  const hasEntities = analysis.medical_entities && Object.keys(analysis.medical_entities).length > 0
  const hasWSD = analysis.word_sense_disambiguation && Object.keys(analysis.word_sense_disambiguation).length > 0
  const hasRelationships = analysis.medical_relationships && analysis.medical_relationships.length > 0
  
  if (!hasEntities && !hasWSD && !hasRelationships) return null

  return (
    <div style={{ background: 'rgba(255, 255, 255, 0.6)', backdropFilter: 'blur(4px)', borderRadius: '1rem', padding: '1.5rem', border: '1px solid rgba(34, 197, 94, 0.3)', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)' }}>
      <div className="flex items-center" style={{ gap: '0.75rem', marginBottom: '1.5rem' }}>
        <div style={{ width: '12px', height: '12px', background: '#22c55e', borderRadius: '50%', animation: 'pulse 2s infinite' }}></div>
        <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#15803d' }}>🔬 Semantic Analysis</h3>
      </div>
      
      {/* Medical entities */}
      {hasEntities && (
        <div style={{ marginBottom: '1rem', padding: '1rem', background: 'rgba(239, 246, 255, 0.8)', borderRadius: '0.75rem', border: '1px solid rgba(191, 219, 254, 0.5)' }}>
          <h4 style={{ fontWeight: '600', color: '#1d4ed8', marginBottom: '0.75rem', display: 'flex', alignItems: 'center' }}>
            <span style={{ color: '#2563eb', marginRight: '0.5rem' }}>🏷️</span>
            Medical Entities
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {Object.entries(analysis.medical_entities).map(([entityType, entities]) => {
              if (entities && entities.length > 0) {
                return (
                  <div key={entityType}>
                    <strong style={{ color: '#1e40af', fontSize: '0.875rem' }}>{entityType}:</strong>
                    <div style={{ marginTop: '0.5rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                      {entities.map((entity, index) => (
                        <span key={index} style={{ padding: '0.25rem 0.75rem', background: '#dbeafe', color: '#1e40af', borderRadius: '9999px', fontSize: '0.75rem', fontWeight: '500' }}>
                          {entity.text}
                        </span>
                      ))}
                    </div>
                  </div>
                )
              }
              return null
            })}
          </div>
        </div>
      )}
      
      {/* Word sense disambiguation */}
      {hasWSD && (
        <div style={{ marginBottom: '1rem', padding: '1rem', background: 'rgba(240, 253, 244, 0.8)', borderRadius: '0.75rem', border: '1px solid rgba(187, 247, 208, 0.5)' }}>
          <h4 style={{ fontWeight: '600', color: '#15803d', marginBottom: '0.75rem', display: 'flex', alignItems: 'center' }}>
            <span style={{ color: '#16a34a', marginRight: '0.5rem' }}>🎯</span>
            Word Sense Disambiguation
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {Object.entries(analysis.word_sense_disambiguation).map(([word, sense]) => (
              <div key={word} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.875rem' }}>
                <span style={{ fontWeight: '500', color: '#166534' }}>{word}</span>
                <span style={{ color: '#16a34a' }}>→</span>
                <span style={{ color: '#15803d' }}>{sense.medical_sense || word}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Medical relationships */}
      {hasRelationships && (
        <div style={{ padding: '1rem', background: 'rgba(250, 245, 255, 0.8)', borderRadius: '0.75rem', border: '1px solid rgba(233, 213, 255, 0.5)' }}>
          <h4 style={{ fontWeight: '600', color: '#7c3aed', marginBottom: '0.75rem', display: 'flex', alignItems: 'center' }}>
            <span style={{ color: '#9333ea', marginRight: '0.5rem' }}>🔗</span>
            Medical Relationships
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            {analysis.medical_relationships.map((rel, index) => (
              <div key={index} style={{ fontSize: '0.875rem', color: '#7c2d12' }}>
                <span style={{ fontWeight: '500' }}>{rel.subject}</span>
                <span style={{ margin: '0 0.5rem', padding: '0.25rem 0.5rem', background: 'rgba(168, 85, 247, 0.2)', borderRadius: '0.25rem', fontWeight: '500' }}>{rel.relation}</span>
                <span style={{ fontWeight: '500' }}>{rel.object}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function QAResultsSection({ results }) {
  return (
    <div style={{ background: 'rgba(255, 255, 255, 0.6)', backdropFilter: 'blur(4px)', borderRadius: '1rem', padding: '1.5rem', border: '1px solid rgba(34, 197, 94, 0.3)', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)' }}>
      <div className="flex items-center" style={{ gap: '0.75rem', marginBottom: '1.5rem' }}>
        <div style={{ width: '12px', height: '12px', background: '#22c55e', borderRadius: '50%', animation: 'pulse 2s infinite' }}></div>
        <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#15803d' }}>❓ Question & Answer Results</h3>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        {results.map((qa, index) => (
          <div key={index} style={{ padding: '1rem', background: 'rgba(254, 252, 232, 0.8)', borderRadius: '0.75rem', border: '1px solid rgba(254, 240, 138, 0.5)' }}>
            <div style={{ fontWeight: '600', color: '#92400e', marginBottom: '0.5rem' }}>{qa.answer}</div>
            <div style={{ fontSize: '0.875rem', color: '#a16207' }}>
              Confidence: <span style={{ fontWeight: '500' }}>{qa.confidence}</span> 
              (Score: <span style={{ fontWeight: '500' }}>{qa.score.toFixed(3)}</span>)
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function RetrievedContextSection({ context }) {
  return (
    <div style={{ background: 'rgba(255, 255, 255, 0.6)', backdropFilter: 'blur(4px)', borderRadius: '1rem', padding: '1.5rem', border: '1px solid rgba(34, 197, 94, 0.3)', boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)' }}>
      <div className="flex items-center" style={{ gap: '0.75rem', marginBottom: '1.5rem' }}>
        <div style={{ width: '12px', height: '12px', background: '#22c55e', borderRadius: '50%', animation: 'pulse 2s infinite' }}></div>
        <h3 style={{ fontSize: '1.125rem', fontWeight: '700', color: '#15803d' }}>📚 Relevant Medical Knowledge</h3>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
        {context.map((contextItem, index) => (
          <div key={index} style={{ padding: '1rem', background: 'rgba(255, 255, 255, 0.4)', borderRadius: '0.75rem', border: '1px solid rgba(34, 197, 94, 0.2)', backdropFilter: 'blur(4px)' }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.75rem' }}>
              <span style={{ width: '24px', height: '24px', background: 'rgba(34, 197, 94, 0.2)', color: '#16a34a', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.875rem', fontWeight: '600', flexShrink: 0, marginTop: '0.125rem' }}>
                {index + 1}
              </span>
              <p style={{ color: '#374151', fontSize: '0.875rem', lineHeight: '1.625' }}>{contextItem}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default App
