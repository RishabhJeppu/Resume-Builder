import sys
import os
import json

sys.path.append("../")
from utils.utils import Utils
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from tqdm import tqdm as stqdm


load_dotenv()


class Groq:
    SYSTEM_PROMPT = "You are a smart assistant to career advisors at the HiDevs. You will reply with JSON only."

    CV_TEXT_PLACEHOLDER = "<CV_TEXT>"
    JD_TEXT_PLACEHOLDER = "<JD_TEXT>"

    SYSTEM_TAILORING = """
        You are a smart assistant to career advisors at the HiDevs. Your take is to rewrite
        resumes to be more brief and convincing according to the Resumes and Cover Letters guide.
    """

    TAILORING_PROMPT = """
        Consider the following CV:
        <CV_TEXT>

        Your task is to rewrite the given CV. Follow these guidelines:
        - Be truthful and objective to the experience listed in the CV
        - Be specific rather than general
        - Rewrite job highlight items using STAR methodology (but do not mention STAR explicitly)
        - Fix spelling and grammar errors
        - Writte to express not impress
        - Articulate and don't be flowery
        - Prefer active voice over passive voice
        - Do not include a summary about the candidate

        Improved CV:
    """

    BASICS_PROMPT = """
        You're an advanced AI developer specializing in extracting structured data from unstructured text. Your expertise lies in creating JSON objects that conform to specific schemas, ensuring that the output is clean and usable. 

        Your task is to write a JSON resume section for an applicant applying for job posts. Here is the TypeScript Interface for the JSON schema: `interface Basics { name: string; email: string; phone: string; website: string; address: string; }` 

        Please extract the personal details from the following text and create a JSON object that adheres to the schema mentioned above. Include only the JSON in your response. 

        Here's the text containing the applicant's details: <CV_TEXT>

        The resulting JSON object should look like this:  
        ```json
        {
        "name": "__________",
        "email": "__________",
        "phone": "__________",
        "linkedin": "__________",
        "address": "__________"
        }
        ``` 

        Make sure to provide a clean and valid JSON output without any additional commentary or formatting.
    """

    EDUCATION_PROMPT = """
        You're an advanced AI developer specializing in extracting structured data from unstructured text. Your expertise lies in creating JSON objects that conform to specific schemas, ensuring that the output is clean and usable. 

        Your task is to write a JSON education section for an applicant applying for job posts. Here is the TypeScript Interface for the JSON schema: 

        ```typescript
        interface EducationItem {
            institution: string;
            area: string;
            additionalAreas: string[];
            studyType: string;
            startDate: string;
            endDate: string;
            score: string;
            location: string;
        }

        interface Education {
            education: EducationItem[];
        }
        ```

        Please extract the education details from the following text and create a JSON object that adheres to the schema mentioned above. Include only the JSON in your response.

        Here's the text containing the applicant's details: <CV_TEXT>

        The resulting JSON object should look like this:  
        ```json
        {
        "education": [
            {
                "institution": "__________",
                "area": "__________",
                "additionalAreas": ["__________", "__________"],
                "studyType": "__________",
                "startDate": "__________",
                "endDate": "__________",
                "score": "__________",
                "location": "__________"
            }
        ]
        }
        ```

        Make sure to provide a clean and valid JSON output without any additional commentary or formatting.
    """

    CERTIFICATES_PROMPT = """
        You're an advanced AI developer specializing in extracting structured data from unstructured text. Your expertise lies in creating JSON objects that conform to specific schemas, ensuring that the output is clean and usable. 

        Your task is to write a JSON awards section for an applicant applying for job posts. Here is the TypeScript Interface for the JSON schema: 

        ```typescript
        interface AwardItem {
            title: string;
            date: string;
            awarder: string;
            summary: string;
        }

        interface Certificates {
            awards: AwardItem[];
        }
        ```

        Please extract the certifications details from the following text and create a JSON object that adheres to the schema mentioned above. Include only the JSON in your response.

        Here's the text containing the applicant's details: <CV_TEXT>

        The resulting JSON object should look like this:  
        ```json
        {
        "awards": [
            {
                "title": "__________",
                "date": "__________",
                "awarder": "__________",
                "summary": "__________"
            }
        ]
        }
        ```

        Make sure to provide a clean and valid JSON output without any additional commentary or formatting.
    """

    PROJECTS_PROMPT = """
        You're an advanced AI developer specializing in extracting structured data from unstructured text. Your expertise lies in creating JSON objects that conform to specific schemas, ensuring that the output is clean and usable.

        Your task is to write a JSON projects section for an applicant applying for job posts. Here is the TypeScript Interface for the JSON schema: 

        ```typescript
        interface ProjectItem {
            name: string;
            description: string;
            keywords: string[];
            url: string;
        }

        interface Projects {
            projects: ProjectItem[];
        }
        ```

        Please extract the project details from the following text and create a JSON object that adheres to the schema mentioned above. Include only the JSON in your response.

        Here's the text containing the applicant's details: <CV_TEXT>

        The resulting JSON object should look like this:  
        ```json
        {
            "projects": [
                {
                    "name": "__________",
                    "description": "__________",
                    "keywords": ["__________", "__________"],
                    "url": "__________"
                },
                {
                    "name": "__________",
                    "description": "__________",
                    "keywords": ["__________", "__________"],
                    "url": "__________"
                }
            ]
        }
        ```

        Ensure that the JSON output includes only the projects extracted from the provided text and is formatted correctly without any additional commentary or formatting.
    """

    SKILLS_PROMPT = """
        You're an advanced AI developer specializing in extracting structured data from unstructured text. Your expertise lies in creating JSON objects that conform to specific schemas, ensuring that the output is clean and usable.

        Your task is to write a JSON skills section for an applicant's resume based on their education and work experience. Here is the TypeScript Interface for the JSON schema:

        ```typescript
        type HardSkills = "Programming Languages" | "Tools" | "Frameworks" | "Computer Proficiency";
        type SoftSkills = "Team Work" | "Communication" | "Leadership" | "Problem Solving" | "Creativity";
        type OtherSkills = string;

        interface SkillItem {
            name: HardSkills | SoftSkills | OtherSkills;
            keywords: string[];
        }

        interface Skills {
            skills: SkillItem[];
        }
        ```

        Please extract the relevant skills from the following text and create a JSON object that adheres to the schema mentioned above. Include only the JSON in your response. Ensure to select the top 4 skill names from the CV that are most relevant to the provided job description.

        Here is the job description: <JD_TEXT>

        Here is the text containing the applicant's details: <CV_TEXT>

        The resulting JSON object should look like this:  
        ```json
        {
            "skills": [
                {
                    "name": "__________",
                    "keywords": ["__________", "__________"]
                },
                {
                    "name": "__________",
                    "keywords": ["__________", "__________"]
                },
                {
                    "name": "__________",
                    "keywords": ["__________", "__________"]
                },
                {
                    "name": "__________",
                    "keywords": ["__________", "__________"]
                }
            ]
        }
        ```

        Ensure that the JSON output includes only the skills extracted from the provided text and is formatted correctly without any additional commentary or formatting.
    """

    WORK_PROMPT = """
        You're an advanced AI developer specializing in extracting structured data from unstructured text. Your expertise lies in creating JSON objects that conform to specific schemas, ensuring that the output is clean and usable.

        Your task is to write a JSON work experience section for an applicant applying for job posts. Here is the TypeScript Interface for the JSON schema:

        ```typescript
        interface WorkItem {
            company: string;
            position: string;
            startDate: string;
            endDate: string;
            location: string;
            highlights: string[];
        }

        interface Work {
            work: WorkItem[];
        }
        ```

        Please extract the relevant work experience from the following text and create a JSON object that adheres to the schema mentioned above. Include only the JSON in your response. Make sure to include the company name, position name, start and end date, location, and highlights for each work experience. 

        Follow the Harvard Extension School Resume guidelines and phrase the highlights using the STAR methodology.

        Here's the job description: <JD_TEXT>

        Here's the text containing the applicant's details: <CV_TEXT>

        If the candidate does not have any work experience, the JSON object should include an empty work array, like this:
        ```json
        {
            "work": []
        }
        ```

        Otherwise, the resulting JSON object should look like this:  
        ```json
        {
            "work": [
                {
                    "company": "__________",
                    "position": "__________",
                    "startDate": "__________",
                    "endDate": "__________",
                    "location": "__________",
                    "highlights": [
                        "__________",
                        "__________"
                    ]
                },
                {
                    "company": "__________",
                    "position": "__________",
                    "startDate": "__________",
                    "endDate": "__________",
                    "location": "__________",
                    "highlights": [
                        "__________",
                        "__________"
                    ]
                }
            ]
        }
        ```

        Ensure that the JSON output includes only the work experience extracted from the provided text. If no experience is found, provide the empty work array as specified above. The output must be formatted correctly without any additional commentary or explanation.
    """

    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-70b-versatile",
        )

    def generate_json_resume(self, jd_text):
        util = Utils()
        dirs = os.listdir("../utils/uploads")[0]
        cv_text = util.extract_text(f"../utils/uploads/{dirs}")
        final_json = {}
        resume_sections = [
            {"prompt": self.BASICS_PROMPT, "key": "basics"},
            {"prompt": self.EDUCATION_PROMPT, "key": "education"},
            {"prompt": self.CERTIFICATES_PROMPT, "key": "awards"},
            {"prompt": self.PROJECTS_PROMPT, "key": "projects"},
            {"prompt": self.SKILLS_PROMPT, "key": "skills"},
            {"prompt": self.WORK_PROMPT, "key": "work"},
        ]

        for section in stqdm(resume_sections, desc="Processing resume sections..."):
            try:
                # Replace placeholders
                filled_cv_prompt = section["prompt"].replace(
                    self.CV_TEXT_PLACEHOLDER, cv_text
                )
                filled_jd_prompt = section["prompt"].replace(
                    self.JD_TEXT_PLACEHOLDER, jd_text
                )

                # Invoke LLM
                response = self.llm.invoke(
                    [
                        SystemMessage(content=self.SYSTEM_PROMPT),
                        HumanMessage(content=filled_cv_prompt),
                        HumanMessage(content=filled_jd_prompt),
                    ]
                )

                # Parse response
                try:
                    answer = json.loads(response.content)
                except json.JSONDecodeError:
                    print(f"Invalid JSON for {section['key']} section")
                    continue

                # Ensure section is wrapped correctly
                if section["key"] not in answer:
                    answer = {section["key"]: answer}

                # Update final JSON
                final_json.update(answer)

            except Exception as e:
                print(f"Error processing {section['key']} section: {e}")

        print(final_json)

    def generate_gmail_message(self):
        # prompt_extract = PromptTemplate.from_template()
        pass

    def generate_linkedin_message(self):
        # promt_extract = PromptTemplate.from_template()
        pass


llm = Groq()

json1 = llm.generate_json_resume(
    """Job description
This is a remote internship role for an Artificial Intelligence Intern
As an AI Intern, you will be responsible for assisting in the development and implementation of AI models and algorithms
You will work on data analysis, machine learning, programming, and other tasks related to AI development
This internship will provide you with valuable hands-on experience in the field of artificial intelligence
Job Requirement
Computer Science and Programming skills
Analytical Skills and Data Science knowledge
Experience or knowledge in Machine Learning
Excellent problem-solving and critical thinking abilities
Strong communication and teamwork skills
Ability to work independently and remotely
Experience with AI frameworks and tools is a plus
Currently pursuing or recently completed a degree in Computer Science, Data Science, or a related field
Role: Data Science & Machine Learning - Other
Industry Type: Internet
Department: Data Science & Analytics
Employment Type: Full Time, Permanent
Role Category: Data Science & Machine Learning
Education
UG: Any Graduate
PG: Any Postgraduate""",
)

print(json1)
