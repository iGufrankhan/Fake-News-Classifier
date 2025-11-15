#!/usr/bin/env python3
"""
📰 FAKE NEWS CLASSIFIER - Interactive README Terminal
Beautiful documentation viewer with navigation
"""

import os
import sys
import time

# ANSI Color Codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BG_BLACK = '\033[40m'
    BG_BLUE = '\033[44m'

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_centered(text, color=''):
    """Print centered text"""
    term_width = os.get_terminal_size().columns
    lines = text.split('\n')
    for line in lines:
        padding = (term_width - len(line)) // 2
        print(' ' * padding + color + line + Colors.ENDC)

def print_box(text, color=Colors.CYAN):
    """Print text in a box"""
    lines = text.split('\n')
    max_len = max(len(line) for line in lines)
    
    print(f"\n{color}╔{'═' * (max_len + 2)}╗{Colors.ENDC}")
    for line in lines:
        padding = max_len - len(line)
        print(f"{color}║{Colors.ENDC} {Colors.BOLD}{line}{Colors.ENDC}{' ' * padding} {color}║{Colors.ENDC}")
    print(f"{color}╚{'═' * (max_len + 2)}╝{Colors.ENDC}")

def print_header():
    """Display main header"""
    header = f"""
{Colors.CYAN}{Colors.BOLD}
    ███████╗ █████╗ ██╗  ██╗███████╗    ███╗   ██╗███████╗██╗    ██╗███████╗
    ██╔════╝██╔══██╗██║ ██╔╝██╔════╝    ████╗  ██║██╔════╝██║    ██║██╔════╝
    █████╗  ███████║█████╔╝ █████╗      ██╔██╗ ██║█████╗  ██║ █╗ ██║███████╗
    ██╔══╝  ██╔══██║██╔═██╗ ██╔══╝      ██║╚██╗██║██╔══╝  ██║███╗██║╚════██║
    ██║     ██║  ██║██║  ██╗███████╗    ██║ ╚████║███████╗╚███╔███╔╝███████║
    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝    ╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝ ╚══════╝
    
                 ██████╗██╗      █████╗ ███████╗███████╗██╗███████╗██╗███████╗██████╗ 
                ██╔════╝██║     ██╔══██╗██╔════╝██╔════╝██║██╔════╝██║██╔════╝██╔══██╗
                ██║     ██║     ███████║███████╗███████╗██║█████╗  ██║█████╗  ██████╔╝
                ██║     ██║     ██╔══██║╚════██║╚════██║██║██╔══╝  ██║██╔══╝  ██╔══██╗
                ╚██████╗███████╗██║  ██║███████║███████║██║██║     ██║███████╗██║  ██║
                 ╚═════╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝
{Colors.ENDC}
{Colors.YELLOW}              🔍 Machine Learning Powered Misinformation Detection 🔍{Colors.ENDC}
{Colors.GREEN}              ═══════════════════════════════════════════════════{Colors.ENDC}
"""
    print(header)

def show_overview():
    """Display project overview"""
    clear_screen()
    print_header()
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}📋 PROJECT OVERVIEW{Colors.ENDC}")
    print(f"{Colors.BLUE}{'═' * 80}{Colors.ENDC}\n")
    
    content = f"""
{Colors.WHITE}Fake news has become a {Colors.RED}{Colors.BOLD}critical issue{Colors.ENDC}{Colors.WHITE} in the digital age. This project
demonstrates how to build a powerful {Colors.GREEN}text classification model{Colors.ENDC}{Colors.WHITE} for detecting
misinformation using {Colors.YELLOW}Natural Language Processing{Colors.ENDC}{Colors.WHITE} and {Colors.YELLOW}Machine Learning{Colors.ENDC}{Colors.WHITE}.{Colors.ENDC}

{Colors.CYAN}{Colors.BOLD}🎯 Key Features:{Colors.ENDC}
  {Colors.GREEN}✓{Colors.ENDC} Complete ML pipeline from data preprocessing to deployment
  {Colors.GREEN}✓{Colors.ENDC} TF-IDF vectorization for feature extraction
  {Colors.GREEN}✓{Colors.ENDC} Linear Support Vector Classifier (LinearSVC) model
  {Colors.GREEN}✓{Colors.ENDC} Comprehensive evaluation metrics
  {Colors.GREEN}✓{Colors.ENDC} Beautiful visualizations with WordClouds and Confusion Matrix

{Colors.CYAN}{Colors.BOLD}💡 What You'll Learn:{Colors.ENDC}
  {Colors.YELLOW}•{Colors.ENDC} Text preprocessing and cleaning techniques
  {Colors.YELLOW}•{Colors.ENDC} Feature extraction using TF-IDF
  {Colors.YELLOW}•{Colors.ENDC} Training and evaluating ML classifiers
  {Colors.YELLOW}•{Colors.ENDC} Model performance visualization
  {Colors.YELLOW}•{Colors.ENDC} Building end-to-end NLP applications
"""
    print(content)
    
    print(f"\n{Colors.BLUE}{'═' * 80}{Colors.ENDC}")
    input(f"\n{Colors.YELLOW}Press ENTER to continue...{Colors.ENDC}")

def show_dataset():
    """Display dataset information"""
    clear_screen()
    print_header()
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}📊 DATASET INFORMATION{Colors.ENDC}")
    print(f"{Colors.BLUE}{'═' * 80}{Colors.ENDC}\n")
    
    content = f"""
{Colors.WHITE}The dataset contains labeled news articles, with each entry classified as either
{Colors.RED}{Colors.BOLD}FAKE{Colors.ENDC}{Colors.WHITE} or {Colors.GREEN}{Colors.BOLD}REAL{Colors.ENDC}{Colors.WHITE}.{Colors.ENDC}

{Colors.CYAN}{Colors.BOLD}📝 Sample Structure:{Colors.ENDC}

{Colors.YELLOW}┌─────────────────────────────┬──────────────────────────────────────┬────────┐
│ Title                       │ Text                                 │ Label  │
├─────────────────────────────┼──────────────────────────────────────┼────────┤{Colors.ENDC}
{Colors.WHITE}│ Donald Trump Sends Out...   │ Donald Trump has reportedly sent...  │{Colors.ENDC} {Colors.RED}FAKE{Colors.ENDC}   {Colors.WHITE}│
│ The economy is improving... │ Reports indicate that the economy... │{Colors.ENDC} {Colors.GREEN}REAL{Colors.ENDC}   {Colors.WHITE}│
│ Scientists discover cure... │ A team of researchers has found...   │{Colors.ENDC} {Colors.GREEN}REAL{Colors.ENDC}   {Colors.WHITE}│
│ Miracle weight loss trick...│ This one weird trick will help...    │{Colors.ENDC} {Colors.RED}FAKE{Colors.ENDC}   {Colors.WHITE}│{Colors.ENDC}
{Colors.YELLOW}└─────────────────────────────┴──────────────────────────────────────┴────────┘{Colors.ENDC}

{Colors.CYAN}{Colors.BOLD}📌 Dataset Details:{Colors.ENDC}
  {Colors.GREEN}•{Colors.ENDC} Source: Kaggle Fake and Real News Dataset
  {Colors.GREEN}•{Colors.ENDC} Format: CSV files with labeled articles
  {Colors.GREEN}•{Colors.ENDC} Features: Title, Text content, Label
  {Colors.GREEN}•{Colors.ENDC} Classes: Binary classification (FAKE/REAL)
  {Colors.GREEN}•{Colors.ENDC} Size: Thousands of news articles

{Colors.YELLOW}{Colors.BOLD}🔗 Dataset Link:{Colors.ENDC}
  {Colors.BLUE}{Colors.UNDERLINE}https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset{Colors.ENDC}
"""
    print(content)
    
    print(f"\n{Colors.BLUE}{'═' * 80}{Colors.ENDC}")
    input(f"\n{Colors.YELLOW}Press ENTER to continue...{Colors.ENDC}")

def show_workflow():
    """Display project workflow"""
    clear_screen()
    print_header()
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}🔄 PROJECT WORKFLOW{Colors.ENDC}")
    print(f"{Colors.BLUE}{'═' * 80}{Colors.ENDC}\n")
    
    steps = [
        ("🧹 DATA PREPROCESSING", [
            "Expand contractions (can't → cannot)",
            "Remove punctuation and special characters",
            "Convert text to lowercase",
            "Remove stopwords",
            "Apply lemmatization"
        ]),
        ("📐 FEATURE EXTRACTION", [
            "Apply TfidfVectorizer",
            "Convert text to numerical features",
            "Max features: 5000 words",
            "Use English stop words"
        ]),
        ("🧠 MODEL TRAINING", [
            "Initialize Linear SVC classifier",
            "Split data (80% train, 20% test)",
            "Fit model on training data",
            "Optimize hyperparameters"
        ]),
        ("📊 MODEL EVALUATION", [
            "Calculate accuracy score",
            "Generate confusion matrix",
            "Compute precision, recall, F1-score",
            "Analyze classification report"
        ]),
        ("🎨 VISUALIZATION", [
            "Create WordClouds for FAKE news",
            "Create WordClouds for REAL news",
            "Plot confusion matrix heatmap",
            "Display performance metrics"
        ])
    ]
    
    for i, (title, items) in enumerate(steps, 1):
        print(f"\n{Colors.GREEN}{Colors.BOLD}Step {i}: {title}{Colors.ENDC}")
        print(f"{Colors.YELLOW}{'─' * 60}{Colors.ENDC}")
        for item in items:
            print(f"  {Colors.CYAN}▸{Colors.ENDC} {Colors.WHITE}{item}{Colors.ENDC}")
    
    print(f"\n{Colors.BLUE}{'═' * 80}{Colors.ENDC}")
    input(f"\n{Colors.YELLOW}Press ENTER to continue...{Colors.ENDC}")

def show_tech_stack():
    """Display technology stack"""
    clear_screen()
    print_header()
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}🛠️  TECHNOLOGY STACK{Colors.ENDC}")
    print(f"{Colors.BLUE}{'═' * 80}{Colors.ENDC}\n")
    
    content = f"""
{Colors.GREEN}{Colors.BOLD}Machine Learning & NLP:{Colors.ENDC}
  {Colors.CYAN}◆{Colors.ENDC} {Colors.YELLOW}scikit-learn{Colors.ENDC} - ML algorithms and tools
    {Colors.WHITE}├─{Colors.ENDC} LinearSVC - Support Vector Classification
    {Colors.WHITE}├─{Colors.ENDC} TfidfVectorizer - Text feature extraction
    {Colors.WHITE}└─{Colors.ENDC} Metrics - accuracy_score, confusion_matrix, classification_report

{Colors.GREEN}{Colors.BOLD}Data Processing:{Colors.ENDC}
  {Colors.CYAN}◆{Colors.ENDC} {Colors.YELLOW}NumPy{Colors.ENDC} - Numerical computing
  {Colors.CYAN}◆{Colors.ENDC} {Colors.YELLOW}Pandas{Colors.ENDC} - Data manipulation and analysis

{Colors.GREEN}{Colors.BOLD}Visualization:{Colors.ENDC}
  {Colors.CYAN}◆{Colors.ENDC} {Colors.YELLOW}Matplotlib{Colors.ENDC} - Creating static plots
  {Colors.CYAN}◆{Colors.ENDC} {Colors.YELLOW}Seaborn{Colors.ENDC} - Statistical data visualization
  {Colors.CYAN}◆{Colors.ENDC} {Colors.YELLOW}WordCloud{Colors.ENDC} - Generating word clouds
  {Colors.CYAN}◆{Colors.ENDC} {Colors.YELLOW}Pillow{Colors.ENDC} - Image processing

{Colors.GREEN}{Colors.BOLD}Development Environment:{Colors.ENDC}
  {Colors.CYAN}◆{Colors.ENDC} {Colors.YELLOW}Jupyter Notebook{Colors.ENDC} - Interactive development
  {Colors.CYAN}◆{Colors.ENDC} {Colors.YELLOW}Python 3.x{Colors.ENDC} - Programming language
"""
    print(content)
    
    print_box("💻 All libraries are open-source and well-documented", Colors.MAGENTA)
    
    print(f"\n{Colors.BLUE}{'═' * 80}{Colors.ENDC}")
    input(f"\n{Colors.YELLOW}Press ENTER to continue...{Colors.ENDC}")

def show_installation():
    """Display installation instructions"""
    clear_screen()
    print_header()
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}⚙️  INSTALLATION & SETUP{Colors.ENDC}")
    print(f"{Colors.BLUE}{'═' * 80}{Colors.ENDC}\n")
    
    print(f"{Colors.GREEN}{Colors.BOLD}Step 1: Clone the Repository{Colors.ENDC}")
    print(f"{Colors.YELLOW}┌────────────────────────────────────────────────────────────────┐")
    print(f"│ {Colors.WHITE}git clone https://github.com/yourusername/fake-news-classifier{Colors.ENDC} {Colors.YELLOW}│")
    print(f"│ {Colors.WHITE}cd fake-news-classifier{Colors.ENDC}                                        {Colors.YELLOW}│")
    print(f"└────────────────────────────────────────────────────────────────┘{Colors.ENDC}\n")
    
    print(f"{Colors.GREEN}{Colors.BOLD}Step 2: Create Virtual Environment (Optional but Recommended){Colors.ENDC}")
    print(f"{Colors.YELLOW}┌────────────────────────────────────────────────────────────────┐")
    print(f"│ {Colors.WHITE}python -m venv venv{Colors.ENDC}                                            {Colors.YELLOW}│")
    print(f"│ {Colors.WHITE}source venv/bin/activate  # On Windows: venv\\Scripts\\activate{Colors.ENDC}  {Colors.YELLOW}│")
    print(f"└────────────────────────────────────────────────────────────────┘{Colors.ENDC}\n")
    
    print(f"{Colors.GREEN}{Colors.BOLD}Step 3: Install Dependencies{Colors.ENDC}")
    print(f"{Colors.YELLOW}┌────────────────────────────────────────────────────────────────┐")
    print(f"│ {Colors.WHITE}pip install numpy pandas scikit-learn{Colors.ENDC}                         {Colors.YELLOW}│")
    print(f"│ {Colors.WHITE}pip install matplotlib seaborn wordcloud pillow{Colors.ENDC}               {Colors.YELLOW}│")
    print(f"│ {Colors.WHITE}pip install jupyter{Colors.ENDC}                                           {Colors.YELLOW}│")
    print(f"└────────────────────────────────────────────────────────────────┘{Colors.ENDC}\n")
    
    print(f"{Colors.GREEN}{Colors.BOLD}Step 4: Download Dataset{Colors.ENDC}")
    print(f"{Colors.WHITE}  • Visit: {Colors.BLUE}{Colors.UNDERLINE}https://www.kaggle.com/datasets{Colors.ENDC}")
    print(f"{Colors.WHITE}  • Download the Fake and Real News dataset{Colors.ENDC}")
    print(f"{Colors.WHITE}  • Place CSV files in the project directory{Colors.ENDC}\n")
    
    print(f"{Colors.GREEN}{Colors.BOLD}Step 5: Run the Notebook{Colors.ENDC}")
    print(f"{Colors.YELLOW}┌────────────────────────────────────────────────────────────────┐")
    print(f"│ {Colors.WHITE}jupyter notebook fake_news_classifier.ipynb{Colors.ENDC}                   {Colors.YELLOW}│")
    print(f"└────────────────────────────────────────────────────────────────┘{Colors.ENDC}\n")
    
    print(f"\n{Colors.BLUE}{'═' * 80}{Colors.ENDC}")
    input(f"\n{Colors.YELLOW}Press ENTER to continue...{Colors.ENDC}")

def show_future_improvements():
    """Display future improvements"""
    clear_screen()
    print_header()
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}🚀 FUTURE IMPROVEMENTS{Colors.ENDC}")
    print(f"{Colors.BLUE}{'═' * 80}{Colors.ENDC}\n")
    
    improvements = [
        ("🤖 Alternative ML Models", [
            "PassiveAggressiveClassifier",
            "Logistic Regression",
            "Naive Bayes",
            "Random Forest Classifier",
            "Gradient Boosting"
        ]),
        ("🧠 Deep Learning Models", [
            "LSTM (Long Short-Term Memory)",
            "BERT (Bidirectional Encoder Representations)",
            "DistilBERT (Lightweight BERT)",
            "RoBERTa (Robustly Optimized BERT)",
            "GPT-based classifiers"
        ]),
        ("🌐 Web Deployment", [
            "Streamlit web application",
            "Flask REST API",
            "Django full-stack app",
            "FastAPI for high performance",
            "Docker containerization"
        ]),
        ("🔗 Advanced Features", [
            "Real-time news classification API",
            "Multi-language support",
            "Browser extension integration",
            "Mobile app development",
            "Ensemble model combination"
        ]),
        ("📈 Enhanced Analytics", [
            "Confidence score display",
            "Explain predictions with LIME/SHAP",
            "A/B testing framework",
            "Model drift monitoring",
            "Performance dashboards"
        ])
    ]
    
    for category, items in improvements:
        print(f"\n{Colors.GREEN}{Colors.BOLD}{category}{Colors.ENDC}")
        print(f"{Colors.YELLOW}{'─' * 60}{Colors.ENDC}")
        for item in items:
            print(f"  {Colors.CYAN}▸{Colors.ENDC} {Colors.WHITE}{item}{Colors.ENDC}")
    
    print(f"\n{Colors.BLUE}{'═' * 80}{Colors.ENDC}")
    input(f"\n{Colors.YELLOW}Press ENTER to continue...{Colors.ENDC}")

def show_performance():
    """Display expected performance metrics"""
    clear_screen()
    print_header()
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}📊 EXPECTED PERFORMANCE METRICS{Colors.ENDC}")
    print(f"{Colors.BLUE}{'═' * 80}{Colors.ENDC}\n")
    
    print(f"{Colors.GREEN}{Colors.BOLD}Model Performance:{Colors.ENDC}\n")
    
    print(f"{Colors.YELLOW}  Accuracy Score:  {Colors.GREEN}{Colors.BOLD}~92-96%{Colors.ENDC}")
    print(f"{Colors.YELLOW}  Precision:       {Colors.GREEN}{Colors.BOLD}~90-95%{Colors.ENDC}")
    print(f"{Colors.YELLOW}  Recall:          {Colors.GREEN}{Colors.BOLD}~90-95%{Colors.ENDC}")
    print(f"{Colors.YELLOW}  F1-Score:        {Colors.GREEN}{Colors.BOLD}~91-95%{Colors.ENDC}\n")
    
    print(f"{Colors.CYAN}{Colors.BOLD}Confusion Matrix (Example):{Colors.ENDC}\n")
    
    print(f"{Colors.WHITE}                Predicted FAKE    Predicted REAL{Colors.ENDC}")
    print(f"{Colors.WHITE}  Actual FAKE   {Colors.GREEN}     950           {Colors.RED}    50{Colors.ENDC}")
    print(f"{Colors.WHITE}  Actual REAL   {Colors.RED}      30           {Colors.GREEN}   970{Colors.ENDC}\n")
    
    print(f"{Colors.CYAN}{Colors.BOLD}Key Insights:{Colors.ENDC}")
    print(f"  {Colors.GREEN}✓{Colors.ENDC} {Colors.WHITE}High accuracy in detecting fake news{Colors.ENDC}")
    print(f"  {Colors.GREEN}✓{Colors.ENDC} {Colors.WHITE}Low false positive rate{Colors.ENDC}")
    print(f"  {Colors.GREEN}✓{Colors.ENDC} {Colors.WHITE}Balanced precision and recall{Colors.ENDC}")
    print(f"  {Colors.GREEN}✓{Colors.ENDC} {Colors.WHITE}Robust to different news topics{Colors.ENDC}")
    
    print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  Note:{Colors.ENDC} {Colors.WHITE}Performance varies based on dataset quality and preprocessing{Colors.ENDC}")
    
    print(f"\n{Colors.BLUE}{'═' * 80}{Colors.ENDC}")
    input(f"\n{Colors.YELLOW}Press ENTER to continue...{Colors.ENDC}")

def show_contributing():
    """Display contribution guidelines"""
    clear_screen()
    print_header()
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}🤝 CONTRIBUTING{Colors.ENDC}")
    print(f"{Colors.BLUE}{'═' * 80}{Colors.ENDC}\n")
    
    print(f"{Colors.WHITE}Contributions are {Colors.GREEN}{Colors.BOLD}welcome{Colors.ENDC}{Colors.WHITE} and greatly appreciated! Here's how you can help:{Colors.ENDC}\n")
    
    print(f"{Colors.GREEN}{Colors.BOLD}1. Fork the Repository{Colors.ENDC}")
    print(f"   {Colors.WHITE}Create your own copy of the project{Colors.ENDC}\n")
    
    print(f"{Colors.GREEN}{Colors.BOLD}2. Create a Feature Branch{Colors.ENDC}")
    print(f"   {Colors.YELLOW}git checkout -b feature/AmazingFeature{Colors.ENDC}\n")
    
    print(f"{Colors.GREEN}{Colors.BOLD}3. Make Your Changes{Colors.ENDC}")
    print(f"   {Colors.WHITE}Add new features, fix bugs, or improve documentation{Colors.ENDC}\n")
    
    print(f"{Colors.GREEN}{Colors.BOLD}4. Commit Your Changes{Colors.ENDC}")
    print(f"   {Colors.YELLOW}git commit -m 'Add some AmazingFeature'{Colors.ENDC}\n")
    
    print(f"{Colors.GREEN}{Colors.BOLD}5. Push to the Branch{Colors.ENDC}")
    print(f"   {Colors.YELLOW}git push origin feature/AmazingFeature{Colors.ENDC}\n")
    
    print(f"{Colors.GREEN}{Colors.BOLD}6. Open a Pull Request{Colors.ENDC}")
    print(f"   {Colors.WHITE}Submit your changes for review{Colors.ENDC}\n")
    
    print(f"{Colors.CYAN}{Colors.BOLD}Areas Where You Can Contribute:{Colors.ENDC}")
    print(f"  {Colors.YELLOW}•{Colors.ENDC} {Colors.WHITE}Improve model accuracy{Colors.ENDC}")
    print(f"  {Colors.YELLOW}•{Colors.ENDC} {Colors.WHITE}Add new visualization features{Colors.ENDC}")
    print(f"  {Colors.YELLOW}•{Colors.ENDC} {Colors.WHITE}Enhance documentation{Colors.ENDC}")
    print(f"  {Colors.YELLOW}•{Colors.ENDC} {Colors.WHITE}Fix bugs and issues{Colors.ENDC}")
    print(f"  {Colors.YELLOW}•{Colors.ENDC} {Colors.WHITE}Add unit tests{Colors.ENDC}")
    print(f"  {Colors.YELLOW}•{Colors.ENDC} {Colors.WHITE}Implement new features{Colors.ENDC}")
    
    print(f"\n{Colors.BLUE}{'═' * 80}{Colors.ENDC}")
    input(f"\n{Colors.YELLOW}Press ENTER to continue...{Colors.ENDC}")

def show_menu():
    """Display main menu"""
    menu_items = [
        ("1", "📋 Project Overview", Colors.CYAN),
        ("2", "📊 Dataset Information", Colors.GREEN),
        ("3", "🔄 Project Workflow", Colors.YELLOW),
        ("4", "🛠️  Technology Stack", Colors.MAGENTA),
        ("5", "⚙️  Installation & Setup", Colors.BLUE),
        ("6", "📊 Performance Metrics", Colors.GREEN),
        ("7", "🚀 Future Improvements", Colors.CYAN),
        ("8", "🤝 Contributing", Colors.YELLOW),
        ("9", "❌ Exit", Colors.RED)
    ]
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}📚 MAIN MENU{Colors.ENDC}")
    print(f"{Colors.BLUE}{'═' * 80}{Colors.ENDC}\n")
    
    for num, title, color in menu_items:
        print(f"  {color}{Colors.BOLD}{num}.{Colors.ENDC} {color}{title}{Colors.ENDC}")
    
    print(f"\n{Colors.BLUE}{'═' * 80}{Colors.ENDC}")

def main():
    """Main application loop"""
    while True:
        clear_screen()
        print_header()
        show_menu()
        
        choice = input(f"\n{Colors.YELLOW}Enter your choice (1-9): {Colors.ENDC}").strip()
        
        if choice == '1':
            show_overview()
        elif choice == '2':
            show_dataset()
        elif choice == '3':
            show_workflow()
        elif choice == '4':
            show_tech_stack()
        elif choice == '5':
            show_installation()
        elif choice == '6':
            show_performance()
        elif choice == '7':
            show_future_improvements()
        elif choice == '8':
            show_contributing()
        elif choice == '9':
            clear_screen()
            print(f"\n{Colors.GREEN}{Colors.BOLD}Thank you for exploring the Fake News Classifier!{Colors.ENDC}")
            print(f"{Colors.CYAN}Stay informed, stay skeptical! 🔍{Colors.ENDC}\n")
            sys.exit(0)
        else:
            print(f"\n{Colors.RED}Invalid choice! Please enter a number between 1-9.{Colors.ENDC}")
            time.sleep(1.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        clear_screen()
        print(f"\n{Colors.YELLOW}Program interrupted by user.{Colors.ENDC}")
        print(f"{Colors.GREEN}Goodbye! 👋{Colors.ENDC}\n")
        sys.exit(0)
