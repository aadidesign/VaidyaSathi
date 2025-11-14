import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  MagnifyingGlassIcon,
  BookOpenIcon,
  AcademicCapIcon,
  DocumentTextIcon
} from '@heroicons/react/24/outline'

const MedicalSearch = () => {
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    
    setIsLoading(true)
    // Simulate search delay
    setTimeout(() => {
      setSearchResults([
        {
          id: 1,
          type: 'research',
          title: 'Clinical Outcomes in Pneumonia Treatment',
          authors: 'Smith J, Johnson B, Chen L',
          journal: 'Journal of Respiratory Medicine',
          year: 2023,
          summary: 'A comprehensive study on modern pneumonia treatment approaches...'
        },
        {
          id: 2,
          type: 'guideline',
          title: 'Evidence-Based Guidelines for Cardiovascular Care',
          organization: 'American Heart Association',
          year: 2023,
          summary: 'Updated guidelines for cardiovascular disease management...'
        }
      ])
      setIsLoading(false)
    }, 1000)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-4xl mx-auto"
    >
      {/* Search Header */}
      <div className="text-center mb-8">
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-3xl font-bold text-neutral-900 mb-4"
        >
          Medical Knowledge Search
        </motion.h1>
        <motion.p
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="text-lg text-neutral-600"
        >
          Search through medical literature, guidelines, and research papers
        </motion.p>
      </div>

      {/* Search Interface */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3 }}
        className="card-premium p-8 mb-8"
      >
        <div className="flex space-x-4">
          <div className="flex-1 relative">
            <MagnifyingGlassIcon className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-neutral-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search medical knowledge, conditions, treatments..."
              className="input-search"
            />
          </div>
          <motion.button
            onClick={handleSearch}
            disabled={isLoading || !searchQuery.trim()}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="btn-primary min-w-[120px]"
          >
            {isLoading ? (
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className="w-5 h-5 border-2 border-white border-t-transparent rounded-full"
              />
            ) : (
              'Search'
            )}
          </motion.button>
        </div>

        {/* Search Filters */}
        <div className="flex flex-wrap gap-2 mt-4">
          {['Research Papers', 'Clinical Guidelines', 'Case Studies', 'Drug Information'].map((filter, index) => (
            <motion.button
              key={filter}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 + index * 0.1 }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-4 py-2 text-sm bg-white/60 hover:bg-white border border-neutral-200 hover:border-primary-300 rounded-lg transition-all duration-200"
            >
              {filter}
            </motion.button>
          ))}
        </div>
      </motion.div>

      {/* Search Results */}
      {searchResults.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          <h2 className="text-2xl font-bold text-neutral-900">Search Results</h2>
          
          {searchResults.map((result, index) => (
            <motion.div
              key={result.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="card-premium p-6 hover:shadow-xl transition-all duration-300"
            >
              <div className="flex items-start space-x-4">
                <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                  result.type === 'research' 
                    ? 'bg-gradient-to-r from-blue-500 to-purple-500' 
                    : 'bg-gradient-to-r from-green-500 to-emerald-500'
                } text-white`}>
                  {result.type === 'research' ? (
                    <AcademicCapIcon className="w-6 h-6" />
                  ) : (
                    <DocumentTextIcon className="w-6 h-6" />
                  )}
                </div>
                
                <div className="flex-1">
                  <h3 className="text-lg font-bold text-neutral-900 mb-2">
                    {result.title}
                  </h3>
                  
                  <div className="text-sm text-neutral-600 mb-3">
                    {result.authors && (
                      <span>{result.authors} • </span>
                    )}
                    {result.organization && (
                      <span>{result.organization} • </span>
                    )}
                    <span>{result.year}</span>
                    {result.journal && (
                      <span> • {result.journal}</span>
                    )}
                  </div>
                  
                  <p className="text-neutral-700 leading-relaxed mb-4">
                    {result.summary}
                  </p>
                  
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className="btn-secondary text-sm"
                  >
                    <BookOpenIcon className="w-4 h-4 mr-2" />
                    Read More
                  </motion.button>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>
      )}

      {/* Empty State */}
      {searchResults.length === 0 && !isLoading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-center py-12"
        >
          <BookOpenIcon className="w-16 h-16 text-neutral-300 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-neutral-600 mb-2">
            Ready to Search
          </h3>
          <p className="text-neutral-500">
            Enter a medical term or condition to search our knowledge base
          </p>
        </motion.div>
      )}
    </motion.div>
  )
}

export default MedicalSearch
