// Design tokens extracted from Figma
// These match the variables defined in your Figma design system

export const designTokens = {
  // Typography
  fonts: {
    primary: 'Inter', // Title Hero/Font Family
  },
  
  fontWeights: {
    heading: 600, // Heading/Font Weight
    semibold: 600, // Single Line/Body Small Strong
  },
  
  fontSizes: {
    bodySmall: '14px', // Single Line/Body Small Strong
    hero: '48px', // From the generated component
  },
  
  lineHeights: {
    tight: 1, // Single Line/Body Small Strong
    hero: 0.95, // From the generated component
  },
  
  // Colors
  colors: {
    text: {
      tertiary: '#b3b3b3', // Text/Default/Tertiary
      primary: '#000000',
      chatHuman: '#ffffff', // Human message text (white on dark)
      chatBot: '#343434', // Bot message text (dark on light)
    },
    background: {
      primary: '#ffffff',
      chatHuman: '#6c6c6c', // Human message bubble (dark gray)
      chatBot: '#e9e9e9', // Bot message bubble (light gray)
    },
    border: {
      default: '#000000',
    }
  },
  
  // Layout
  layout: {
    deviceWidth: 1200, // Device Width
    mobileBreakpoint: 390, // From min-w-[390px] in generated code
  },
  
  // Spacing (extracted from generated component)
  spacing: {
    container: {
      x: '157px', // px-[157px]
      y: '40px', // py-10 (2.5rem = 40px)
    },
    button: {
      x: '29px', // px-[29px]
      y: '12px', // py-3
    },
    input: {
      x: '25px', // left-[25px]
    }
  },
  
  // Border radius
  borderRadius: {
    button: '8px', // rounded-lg
    input: '36px', // rounded-[36px]
  }
} as const

// Tailwind CSS custom classes that match Figma design
export const figmaClasses = {
  heroText: 'text-[48px] font-bold leading-[0.95] tracking-[-1.92px] text-center',
  buttonText: 'text-[14px] font-semibold leading-none text-center',
  inputText: 'text-[14px] font-medium tracking-[-0.56px] text-[#b3b3b3]',
  actionButton: 'bg-white border border-black rounded-lg px-[29px] py-3 opacity-[0.81] hover:opacity-100 transition-opacity',
  chatInput: 'bg-white border border-black rounded-[36px] h-12',
  // Chat interface specific classes
  chatHeader: 'text-[32px] font-semibold leading-[0.95] tracking-[-1.28px]',
  humanMessage: 'bg-[#6c6c6c] text-[#ffffff] text-[14px] font-medium leading-[0.95] tracking-[-0.56px] rounded-[18px] px-[15px] pr-[19px] py-7',
  botMessage: 'bg-[#e9e9e9] text-[#343434] text-[14px] font-medium leading-[0.95] tracking-[-0.56px] rounded-[18px] px-[15px] pr-[19px] py-7',
} as const
