import { motion } from 'framer-motion'
import { SparklesIcon } from '@heroicons/react/24/outline'

const TypingIndicator = () => {
  const dotVariants = {
    initial: { y: 0 },
    animate: { y: -8 },
  }

  const dotTransition = {
    duration: 0.5,
    repeat: Infinity,
    repeatType: "reverse",
    ease: "easeInOut"
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="flex justify-start animate-fade-in"
    >
      <div className="flex items-start space-x-3 max-w-md">
        {/* Avatar */}
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
          className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-xl flex items-center justify-center shadow-lg"
        >
          <SparklesIcon className="w-5 h-5" />
        </motion.div>

        {/* Typing Animation */}
        <div className="bg-gradient-to-r from-white to-neutral-50 border border-neutral-200/50 px-6 py-4 rounded-2xl rounded-bl-md shadow-lg">
          <div className="flex items-center space-x-2">
            <span className="text-sm text-neutral-600 font-medium">AI is thinking</span>
            <div className="flex space-x-1">
              <motion.div
                variants={dotVariants}
                initial="initial"
                animate="animate"
                transition={{ ...dotTransition, delay: 0 }}
                className="w-2 h-2 bg-primary-500 rounded-full"
              />
              <motion.div
                variants={dotVariants}
                initial="initial"
                animate="animate"
                transition={{ ...dotTransition, delay: 0.2 }}
                className="w-2 h-2 bg-primary-400 rounded-full"
              />
              <motion.div
                variants={dotVariants}
                initial="initial"
                animate="animate"
                transition={{ ...dotTransition, delay: 0.4 }}
                className="w-2 h-2 bg-primary-300 rounded-full"
              />
            </div>
          </div>
          
          {/* Processing indicator */}
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: "100%" }}
            transition={{ duration: 2, repeat: Infinity }}
            className="mt-3 h-0.5 bg-gradient-to-r from-primary-500 via-blue-500 to-purple-500 rounded-full"
          />
        </div>
      </div>
    </motion.div>
  )
}

export default TypingIndicator
