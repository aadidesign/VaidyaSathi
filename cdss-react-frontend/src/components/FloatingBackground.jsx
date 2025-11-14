import { motion } from 'framer-motion'

const FloatingBackground = () => {
  const shapes = [
    { size: 300, color: 'from-primary-500/10 to-emerald-500/10', delay: 0 },
    { size: 200, color: 'from-blue-500/10 to-purple-500/10', delay: 2 },
    { size: 150, color: 'from-rose-500/10 to-pink-500/10', delay: 4 },
    { size: 100, color: 'from-yellow-500/10 to-orange-500/10', delay: 1 },
    { size: 250, color: 'from-indigo-500/10 to-cyan-500/10', delay: 3 }
  ]

  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none">
      {shapes.map((shape, index) => (
        <motion.div
          key={index}
          className={`absolute rounded-full bg-gradient-to-r ${shape.color} blur-3xl`}
          style={{
            width: shape.size,
            height: shape.size,
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
          }}
          animate={{
            x: [0, 30, -30, 0],
            y: [0, -30, 30, 0],
            scale: [1, 1.1, 0.9, 1],
          }}
          transition={{
            duration: 8 + Math.random() * 4,
            repeat: Infinity,
            delay: shape.delay,
            ease: "easeInOut"
          }}
        />
      ))}

      {/* Medical pattern overlay */}
      <div className="absolute inset-0 opacity-30">
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="medical-grid" width="100" height="100" patternUnits="userSpaceOnUse">
              <circle cx="50" cy="50" r="2" fill="currentColor" className="text-primary-500/20" />
              <circle cx="25" cy="25" r="1" fill="currentColor" className="text-emerald-500/20" />
              <circle cx="75" cy="75" r="1" fill="currentColor" className="text-blue-500/20" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#medical-grid)" />
        </svg>
      </div>

      {/* Gradient overlays */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary-50/50 via-transparent to-emerald-50/50" />
      <div className="absolute inset-0 bg-gradient-to-tl from-blue-50/30 via-transparent to-purple-50/30" />
    </div>
  )
}

export default FloatingBackground
