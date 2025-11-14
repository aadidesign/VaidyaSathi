# CDSS React Frontend

A premium, modern React frontend for the Clinical Decision Support System (CDSS) built with Vite, Tailwind CSS, and Framer Motion.

## 🌟 Features

- **Premium UI/UX**: Modern, clean, and minimal design with glassmorphism effects
- **Advanced Animations**: Smooth, professional animations using Framer Motion
- **Responsive Design**: Mobile-first approach with responsive layouts
- **Real-time Chat**: Interactive chat interface with the RAG-enhanced AI
- **Medical Search**: Advanced search functionality for medical knowledge
- **Analysis Dashboard**: Comprehensive analysis panel with NLP processing results
- **Component Library**: Reusable, well-structured React components

## 🚀 Tech Stack

- **React 18** - Modern React with hooks
- **Vite** - Fast build tool and development server
- **Tailwind CSS** - Utility-first CSS framework
- **Framer Motion** - Production-ready motion library
- **Heroicons** - Beautiful hand-crafted SVG icons
- **Axios** - Promise-based HTTP client
- **Headless UI** - Unstyled, accessible UI components

## 🎨 Design System

### Color Palette
- **Primary**: Green tones (#22c55e to #14532d)
- **Neutral**: Modern grays (#fafafa to #0a0a0a)
- **Medical**: Blue, purple, rose, amber accents
- **Semantic**: Success, warning, error, info states

### Typography
- **Primary Font**: Inter (300-900 weights)
- **Secondary Font**: Geist (300-700 weights)
- **Font Features**: OpenType features for enhanced readability

### Components
- **Glass Cards**: Glassmorphism with backdrop blur
- **Premium Buttons**: Gradient backgrounds with hover effects
- **Medical Cards**: Specialized cards for diagnosis, recommendations
- **Progress Bars**: Animated progress indicators
- **Message Bubbles**: Chat interface with modern styling

## 📦 Installation

1. **Navigate to the frontend directory:**
   ```bash
   cd cdss-react-frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm run dev
   ```

4. **Open your browser:**
   Visit [http://localhost:3000](http://localhost:3000)

## 🔧 Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run start` - Start server with host access
- `npm run build:production` - Production build with optimizations

### Project Structure

```
cdss-react-frontend/
├── src/
│   ├── components/          # Reusable React components
│   │   ├── Header.jsx      # Main header with status indicators
│   │   ├── ChatInterface.jsx # Chat interface with AI
│   │   ├── MessageBubble.jsx # Individual message components
│   │   ├── TypingIndicator.jsx # AI thinking animation
│   │   ├── AnalysisPanel.jsx # Analysis results panel
│   │   ├── MedicalSearch.jsx # Search functionality
│   │   ├── FeatureGrid.jsx  # Feature showcase grid
│   │   ├── FloatingBackground.jsx # Animated background
│   │   └── ProgressBar.jsx  # Animated progress bars
│   ├── App.jsx             # Main application component
│   ├── main.jsx            # Application entry point
│   └── index.css           # Global styles and utilities
├── public/                 # Static assets
├── tailwind.config.js      # Tailwind configuration
├── vite.config.js         # Vite configuration
└── package.json           # Dependencies and scripts
```

## 🔌 API Integration

The frontend integrates with the Django backend API:

- **Base URL**: `http://127.0.0.1:8000`
- **Chat API**: `/api/rag-chat/` - RAG-enhanced chat
- **Search API**: `/api/medical-knowledge-search/` - Medical search
- **Features API**: `/api/test-all-features/` - Feature testing
- **Health API**: `/api/health/` - Health check

### Proxy Configuration

Vite is configured to proxy API requests to the Django backend:

```javascript
proxy: {
  '/api': {
    target: 'http://127.0.0.1:8000',
    changeOrigin: true,
    secure: false,
  }
}
```

## 🎯 Key Features

### 1. **Premium Chat Interface**
- Real-time messaging with AI
- Typing indicators and animations
- Message history with timestamps
- Quick suggestion buttons

### 2. **Advanced Analysis Panel**
- NLP processing statistics
- Differential diagnoses with confidence scores
- Medical recommendations
- Research paper integration

### 3. **Medical Search**
- Intelligent search with filters
- Research papers and guidelines
- Real-time search results
- Category-based filtering

### 4. **Responsive Design**
- Mobile-first approach
- Tablet and desktop optimized
- Touch-friendly interactions
- Accessible components

### 5. **Modern Animations**
- Smooth page transitions
- Loading states and skeletons
- Hover effects and micro-interactions
- Staggered animations

## 🎨 UI/UX Highlights

- **Glassmorphism**: Modern glass effects with backdrop blur
- **Gradient Designs**: Beautiful color gradients throughout
- **Micro-interactions**: Subtle animations for better UX
- **Loading States**: Professional loading indicators
- **Error Handling**: User-friendly error messages
- **Accessibility**: WCAG compliant components

## 🚀 Production Build

For production deployment:

```bash
npm run build:production
```

This creates optimized bundles with:
- Code splitting for better performance
- Minified and compressed assets
- Vendor chunk separation
- Tree-shaking for smaller bundles

## 🔧 Customization

### Theme Configuration
Modify `tailwind.config.js` to customize:
- Color palette
- Typography scale
- Animation timings
- Breakpoints

### Component Styling
Update `src/index.css` for:
- Component base styles
- Utility classes
- Animation keyframes
- Custom CSS properties

## 📱 Browser Support

- **Modern Browsers**: Chrome, Firefox, Safari, Edge
- **Mobile**: iOS Safari, Chrome Mobile
- **Features**: ES2020+, CSS Grid, Flexbox, CSS Custom Properties

## 🤝 Integration with Django

The React frontend seamlessly integrates with the existing Django CDSS backend:

1. **Preserved Functionality**: All existing API endpoints work unchanged
2. **Enhanced UI**: Modern interface while maintaining all features
3. **Improved UX**: Better user experience with animations and responsiveness
4. **Professional Design**: Premium look and feel for clinical environments

## 📧 Support

For questions or issues with the frontend:
1. Check the browser console for errors
2. Verify the Django backend is running
3. Ensure all dependencies are installed
4. Check network connectivity to the API