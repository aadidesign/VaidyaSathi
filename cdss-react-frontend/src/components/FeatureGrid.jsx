import { motion } from 'framer-motion'

const FeatureGrid = ({ features }) => {
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: {
      opacity: 1,
      y: 0,
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
      className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
    >
      {features.map((feature, index) => (
        <motion.div
          key={index}
          variants={itemVariants}
          whileHover={{ 
            scale: 1.05,
            transition: { type: "spring", stiffness: 400, damping: 10 }
          }}
          className="card-premium p-6 group cursor-pointer"
        >
          <div className="flex items-start space-x-4">
            <motion.div
              whileHover={{ rotate: 15 }}
              className={`w-12 h-12 bg-gradient-to-r ${feature.color} rounded-xl flex items-center justify-center text-white shadow-lg group-hover:shadow-xl transition-shadow duration-300`}
            >
              <feature.icon className="w-6 h-6" />
            </motion.div>
            
            <div className="flex-1">
              <h3 className="text-lg font-bold text-neutral-900 mb-2 group-hover:text-primary-600 transition-colors duration-200">
                {feature.title}
              </h3>
              <p className="text-neutral-600 text-sm leading-relaxed">
                {feature.description}
              </p>
            </div>
          </div>

          {/* Hover effect overlay */}
          <motion.div
            initial={{ opacity: 0 }}
            whileHover={{ opacity: 1 }}
            className="absolute inset-0 bg-gradient-to-r from-primary-500/5 to-emerald-500/5 rounded-2xl -z-10"
          />
        </motion.div>
      ))}
    </motion.div>
  )
}

export default FeatureGrid
