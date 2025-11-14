import { motion } from 'framer-motion'
import { 
  UserIcon, 
  SparklesIcon,
  ExclamationTriangleIcon,
  ClockIcon
} from '@heroicons/react/24/outline'

const MessageBubble = ({ message }) => {
  const isUser = message.type === 'user'
  const isError = message.isError

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    })
  }

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
        <strong key={`bold-${match.index}`} className="font-bold">
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

  const containerVariants = {
    hidden: { opacity: 0, y: 20, scale: 0.8 },
    visible: { 
      opacity: 1, 
      y: 0, 
      scale: 1,
      transition: {
        type: "spring",
        stiffness: 500,
        damping: 30
      }
    }
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} animate-fade-in`}
    >
      <div className={`flex items-start space-x-3 max-w-md lg:max-w-lg ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
        {/* Avatar */}
        <motion.div
          whileHover={{ scale: 1.1 }}
          className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 ${
            isUser 
              ? 'bg-gradient-to-r from-primary-500 to-emerald-500 text-white shadow-medical' 
              : isError
                ? 'bg-gradient-to-r from-red-500 to-orange-500 text-white shadow-lg'
                : 'bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-lg'
          }`}
        >
          {isUser ? (
            <UserIcon className="w-5 h-5" />
          ) : isError ? (
            <ExclamationTriangleIcon className="w-5 h-5" />
          ) : (
            <SparklesIcon className="w-5 h-5" />
          )}
        </motion.div>

        {/* Message Content */}
        <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
          <motion.div
            whileHover={{ scale: 1.02 }}
            className={`px-6 py-4 rounded-2xl shadow-lg ${
              isUser
                ? 'bg-gradient-to-r from-primary-500 to-emerald-500 text-white rounded-br-md'
                : isError
                  ? 'bg-gradient-to-r from-red-50 to-orange-50 text-red-800 border border-red-200 rounded-bl-md'
                  : 'bg-gradient-to-r from-white to-neutral-50 text-neutral-800 border border-neutral-200/50 rounded-bl-md'
            }`}
          >
            {/* Message Text */}
            <div className={`text-sm font-medium leading-relaxed ${isUser ? 'text-white' : isError ? 'text-red-800' : 'text-neutral-800'}`}>
              {message.content.split('\n').map((line, index) => (
                <div key={index}>
                  {parseMarkdown(line)}
                  {index < message.content.split('\n').length - 1 && <br />}
                </div>
              ))}
            </div>

            {/* Analysis Indicator */}
            {message.analysis && !isUser && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
                className="mt-3 pt-3 border-t border-neutral-200/30"
              >
                <div className="flex items-center space-x-2 text-xs text-neutral-600">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    className="w-3 h-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                  />
                  <span>Analysis data available</span>
                </div>
              </motion.div>
            )}
          </motion.div>

          {/* Timestamp */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className={`flex items-center space-x-1 mt-2 text-xs text-neutral-500 ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}
          >
            <ClockIcon className="w-3 h-3" />
            <span>{formatTime(message.timestamp)}</span>
          </motion.div>
        </div>
      </div>
    </motion.div>
  )
}

export default MessageBubble
