from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from agentops.agent import track_agent
import agentops
import os

from crewai_tools import (
    ScrapeWebsiteTool
)

# Set up API keys
# os.environ["SERPER_API_KEY"] = "Your Key" # serper.dev API key
# os.environ["OPENAI_API_KEY"] = "Your Key"

# Instantiate tools
#ScrapeWebsiteTool = ScrapeWebsiteTool()
#WebsiteSearchTool = WebsiteSearchTool()

agentops.init()

@CrewBase
class QaDatasetGeneratorCrew():
    """Python Library Dataset Generation Crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self) -> None:
        """
        Initializes the object with a ChatGroq instance for Groq operations.

        Parameters:
            self: The object instance.
        
        Returns:
            None
        """
        # Groq
        self.groq_llm = ChatGroq(
            temperature=0,
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            #model_name="llama3-70b-8192",
            model_name="gemma-7b-it",
            #model_name="mixtral-8x7b-32768",
        )

    @track_agent(name="web_scraper")
    @agent
    def web_scraper(self) -> Agent:
        """
        Defines the web scraping agent function that returns an Agent object.

        Parameters:
            self: The object instance.

        Returns:
            Agent: An instance of the Agent class.
        """
        return Agent(
            config=self.agents_config['web_scraper'],
            tools=[ScrapeWebsiteTool()],
            llm=self.groq_llm,  
            verbose=True
        )
    @track_agent(name="dataset_builder")
    @agent
    def dataset_builder(self) -> Agent:
        """
        Returns an Agent object based on the dataset_builder configuration, ChatGroq instance, and verbosity setting.
        """
        return Agent(
            config=self.agents_config['dataset_builder'],
            llm=self.groq_llm,
            verbose=True
        )
    @track_agent(name="documentation_analyst")
    @agent
    def documentation_analyst(self) -> Agent:
        """
        Returns an Agent object based on the documentation_analyst configuration, GroqLLM instance, and verbosity setting.
        """
        return Agent(
            config=self.agents_config['documentation_analyst'],
            llm=self.groq_llm,
            verbose=True
        )

    @task
    def extract_library_features(self) -> Task:
        """
        Extracts library features based on the configuration and the web scraping agent.
        """
        return Task(
            config=self.tasks_config['extract_library_features'],
            agent=self.web_scraper()
        )

    @task
    def generate_QA_pairs(self) -> Task:
        """
        Returns a Task object for generating question and answer pairs using the tasks configuration and the dataset builder agent.
        """
        return Task(
            config=self.tasks_config['generate_QA_pairs'],
            agent=self.dataset_builder()
        )

    @task
    def analyze_documentation_insights(self) -> Task:
        """
        Analyzes documentation insights based on the provided configuration and generates a Task object.
        
        Parameters:
            self: The object instance.
        
        Returns:
            Task: An instance of the Task class representing the analysis of documentation insights.
        """
        return Task(
            config=self.tasks_config['analyze_documentation_insights'],
            agent=self.documentation_analyst(),
            output_file='documentation_insights_report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Python Library Dataset Generation Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            memory=False,
            max_rpm=2,
            max_iter=2,
            verbose=2,
        )
