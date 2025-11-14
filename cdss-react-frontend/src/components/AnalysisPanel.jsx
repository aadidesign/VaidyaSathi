import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  ChartBarIcon,
  CpuChipIcon,
  DocumentTextIcon,
  BeakerIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  BookOpenIcon
} from '@heroicons/react/24/outline'
import ProgressBar from './ProgressBar'

const AnalysisPanel = ({ analysisData }) => {
  const [activeSection, setActiveSection] = useState('overview')

  const sections = [
    { id: 'overview', label: 'Overview', icon: ChartBarIcon },
    { id: 'nlp', label: 'NLP Status', icon: CpuChipIcon },
    { id: 'diagnosis', label: 'Diagnosis', icon: DocumentTextIcon },
    { id: 'research', label: 'Research', icon: BookOpenIcon }
  ]

  const mockData = {
    preprocessing_stats: {
      sentence_count: 3,
      token_count: 45,
      normalized_count: 42,
      pos_count: 45,
      spell_corrections: 1
    },
    differential_diagnoses: [
      {
        condition: "Tension Headache",
        confidence: 75,
        explanation: "Common headache type often associated with stress and muscle tension"
      },
      {
        condition: "Migraine",
        confidence: 45,
        explanation: "Severe headache often accompanied by nausea and light sensitivity"
      },
      {
        condition: "Secondary Headache",
        confidence: 25,
        explanation: "Headache due to underlying medical condition requiring investigation"
      }
    ],
    recommendations: {
      immediate_actions: [
        "Consult with healthcare provider as soon as possible",
        "Monitor symptoms and keep detailed log"
      ],
      tests: [
        "Complete blood count (CBC)",
        "Comprehensive metabolic panel",
        "Brain imaging if symptoms persist"
      ]
    }
  }

  const data = analysisData || mockData

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="card-premium overflow-hidden h-[600px] flex flex-col"
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-6 text-white">
        <div className="flex items-center space-x-3">
          <motion.div
            animate={{ scale: [1, 1.1, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center"
          >
            <ChartBarIcon className="w-5 h-5" />
          </motion.div>
          <div>
            <h2 className="text-xl font-bold">Analysis Panel</h2>
            <p className="text-blue-100 text-sm">AI Processing Results</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <div className="border-b border-neutral-200/50 bg-white/50">
        <div className="flex overflow-x-auto">
          {sections.map((section) => (
            <motion.button
              key={section.id}
              onClick={() => setActiveSection(section.id)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className={`flex items-center space-x-2 px-4 py-3 text-sm font-medium border-b-2 transition-all duration-200 whitespace-nowrap ${
                activeSection === section.id
                  ? 'border-primary-500 text-primary-600 bg-primary-50/50'
                  : 'border-transparent text-neutral-600 hover:text-neutral-900 hover:border-neutral-300'
              }`}
            >
              <section.icon className="w-4 h-4" />
              <span>{section.label}</span>
            </motion.button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-6 bg-gradient-to-b from-white/50 to-blue-50/30 scrollbar-hide">
        <AnimatePresence mode="wait">
          {activeSection === 'overview' && (
            <motion.div
              key="overview"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              <WelcomeCard />
              {data.differential_diagnoses && <QuickDiagnosis diagnoses={data.differential_diagnoses} />}
            </motion.div>
          )}

          {activeSection === 'nlp' && (
            <motion.div
              key="nlp"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              <NLPStatusCard stats={data.preprocessing_stats} />
            </motion.div>
          )}

          {activeSection === 'diagnosis' && (
            <motion.div
              key="diagnosis"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {data.differential_diagnoses && <DiagnosisCard diagnoses={data.differential_diagnoses} />}
              {data.recommendations && <RecommendationsCard recommendations={data.recommendations} />}
            </motion.div>
          )}

          {activeSection === 'research' && (
            <motion.div
              key="research"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              <ResearchCard />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  )
}

const WelcomeCard = () => (
  <div className="medical-card">
    <div className="flex items-start space-x-3">
      <div className="w-3 h-3 bg-primary-500 rounded-full animate-pulse mt-1" />
      <div>
        <h3 className="text-lg font-bold text-primary-700 mb-2">Welcome</h3>
        <p className="text-neutral-700 leading-relaxed">
          Detailed analysis of your query will appear here after you send a message.
        </p>
      </div>
    </div>
  </div>
)

const QuickDiagnosis = ({ diagnoses }) => (
  <div className="diagnosis-card">
    <h3 className="text-lg font-bold text-blue-700 mb-4 flex items-center">
      <DocumentTextIcon className="w-5 h-5 mr-2" />
      Quick Diagnosis
    </h3>
    <div className="space-y-3">
      {diagnoses.slice(0, 2).map((diagnosis, index) => (
        <div key={index} className="flex items-center justify-between">
          <span className="text-sm font-medium text-blue-800">{diagnosis.condition}</span>
          <span className="text-xs text-blue-600 font-semibold">{diagnosis.confidence}%</span>
        </div>
      ))}
    </div>
  </div>
)

const NLPStatusCard = ({ stats }) => (
  <div className="medical-card">
    <h3 className="text-lg font-bold text-purple-700 mb-4 flex items-center">
      <CpuChipIcon className="w-5 h-5 mr-2" />
      NLP Processing Status
    </h3>
    {stats ? (
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-purple-50 rounded-lg p-3">
          <div className="text-2xl font-bold text-purple-700">{stats.sentence_count}</div>
          <div className="text-xs text-purple-600">Sentences</div>
        </div>
        <div className="bg-purple-50 rounded-lg p-3">
          <div className="text-2xl font-bold text-purple-700">{stats.token_count}</div>
          <div className="text-xs text-purple-600">Tokens</div>
        </div>
        <div className="bg-purple-50 rounded-lg p-3">
          <div className="text-2xl font-bold text-purple-700">{stats.normalized_count}</div>
          <div className="text-xs text-purple-600">Normalized</div>
        </div>
        <div className="bg-purple-50 rounded-lg p-3">
          <div className="text-2xl font-bold text-purple-700">{stats.spell_corrections}</div>
          <div className="text-xs text-purple-600">Corrections</div>
        </div>
      </div>
    ) : (
      <p className="text-neutral-600">Processing stats will appear after analysis.</p>
    )}
  </div>
)

const DiagnosisCard = ({ diagnoses }) => (
  <div className="medical-card">
    <h3 className="text-lg font-bold text-green-700 mb-4 flex items-center">
      <BeakerIcon className="w-5 h-5 mr-2" />
      Differential Diagnoses
    </h3>
    <div className="space-y-4">
      {diagnoses.map((diagnosis, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className="bg-green-50 rounded-xl p-4"
        >
          <div className="flex justify-between items-start mb-2">
            <h4 className="font-semibold text-green-800">{diagnosis.condition}</h4>
            <span className="text-sm font-bold text-green-600">{diagnosis.confidence}%</span>
          </div>
          <ProgressBar percentage={diagnosis.confidence} color="green" />
          <p className="text-sm text-green-700 mt-2 leading-relaxed">{diagnosis.explanation}</p>
        </motion.div>
      ))}
    </div>
  </div>
)

const RecommendationsCard = ({ recommendations }) => (
  <div className="recommendation-card">
    <h3 className="text-lg font-bold text-red-700 mb-4 flex items-center">
      <ExclamationTriangleIcon className="w-5 h-5 mr-2" />
      Recommendations
    </h3>
    
    {recommendations.immediate_actions && (
      <div className="mb-4">
        <h4 className="font-semibold text-red-800 mb-2 flex items-center">
          <ClockIcon className="w-4 h-4 mr-1" />
          Immediate Actions
        </h4>
        <ul className="space-y-1">
          {recommendations.immediate_actions.map((action, index) => (
            <li key={index} className="text-sm text-red-700 flex items-start">
              <div className="w-1.5 h-1.5 bg-red-500 rounded-full mt-2 mr-2 flex-shrink-0" />
              {action}
            </li>
          ))}
        </ul>
      </div>
    )}

    {recommendations.tests && (
      <div>
        <h4 className="font-semibold text-blue-800 mb-2 flex items-center">
          <BeakerIcon className="w-4 h-4 mr-1" />
          Recommended Tests
        </h4>
        <ul className="space-y-1">
          {recommendations.tests.map((test, index) => (
            <li key={index} className="text-sm text-blue-700 flex items-start">
              <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2 mr-2 flex-shrink-0" />
              {test}
            </li>
          ))}
        </ul>
      </div>
    )}
  </div>
)

const ResearchCard = () => (
  <div className="research-card">
    <h3 className="text-lg font-bold text-purple-700 mb-4 flex items-center">
      <BookOpenIcon className="w-5 h-5 mr-2" />
      Research Papers
    </h3>
    <p className="text-purple-600 text-sm">
      Relevant research papers will appear here based on your analysis.
    </p>
  </div>
)

export default AnalysisPanel
