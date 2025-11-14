import { motion } from 'framer-motion'
import { 
  HeartIcon, 
  SparklesIcon,
  CpuChipIcon,
  BeakerIcon
} from '@heroicons/react/24/outline'

const Header = () => {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative z-20"
    >
      <div className="glass-dark backdrop-blur-xl border-b border-white/10">
        <div className="medical-layout py-6">
          <div className="flex items-center justify-between">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="flex items-center space-x-4"
            >
              {/* Logo */}
              <div className="relative">
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                  className="w-12 h-12 bg-gradient-to-r from-primary-500 to-emerald-500 rounded-2xl flex items-center justify-center shadow-lg"
                >
                  <HeartIcon className="w-6 h-6 text-white" />
                </motion.div>
                <motion.div
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="absolute -top-1 -right-1 w-4 h-4 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center"
                >
                  <SparklesIcon className="w-2 h-2 text-white" />
                </motion.div>
              </div>

              {/* Title */}
              <div>
                <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-white to-primary-100 bg-clip-text text-transparent">
                  CDSS Platform
                </h1>
                <p className="text-sm text-primary-100/80 font-medium">
                  Clinical Decision Support System
                </p>
              </div>
            </motion.div>

            {/* Status Indicators */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="flex items-center space-x-4"
            >
              {/* AI Status */}
              <div className="glass rounded-xl px-4 py-2 flex items-center space-x-2">
                <motion.div
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="w-2 h-2 bg-green-400 rounded-full"
                />
                <CpuChipIcon className="w-4 h-4 text-white/80" />
                <span className="text-xs text-white/80 font-medium">AI Online</span>
              </div>

              {/* Knowledge Base Status */}
              <div className="glass rounded-xl px-4 py-2 flex items-center space-x-2">
                <motion.div
                  animate={{ scale: [1, 1.1, 1] }}
                  transition={{ duration: 2, repeat: Infinity, delay: 0.5 }}
                  className="w-2 h-2 bg-blue-400 rounded-full"
                />
                <BeakerIcon className="w-4 h-4 text-white/80" />
                <span className="text-xs text-white/80 font-medium">RAG Active</span>
              </div>
            </motion.div>
          </div>

          {/* Subtitle */}
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="mt-4 text-center"
          >
            <p className="text-primary-100/70 text-lg font-medium">
              AI-Powered Clinical Intelligence for Enhanced Patient Care
            </p>
          </motion.div>
        </div>
      </div>

      {/* Animated border */}
      <div className="h-px bg-gradient-to-r from-transparent via-primary-400/50 to-transparent" />
    </motion.header>
  )
}

export default Header
