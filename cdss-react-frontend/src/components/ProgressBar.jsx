import { motion } from 'framer-motion'

const ProgressBar = ({ percentage, color = 'primary', className = '' }) => {
  const colorClasses = {
    primary: 'from-primary-500 to-primary-600',
    green: 'from-green-500 to-green-600',
    blue: 'from-blue-500 to-blue-600',
    red: 'from-red-500 to-red-600',
    purple: 'from-purple-500 to-purple-600',
    yellow: 'from-yellow-500 to-yellow-600'
  }

  return (
    <div className={`progress-bar ${className}`}>
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${percentage}%` }}
        transition={{ duration: 1, ease: "easeOut" }}
        className={`progress-fill bg-gradient-to-r ${colorClasses[color]} relative overflow-hidden`}
      >
        {/* Shimmer effect */}
        <motion.div
          animate={{ x: ['100%', '-100%'] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
          className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent"
        />
      </motion.div>
    </div>
  )
}

export default ProgressBar
