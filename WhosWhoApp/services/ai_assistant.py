import os
import concurrent.futures
from huggingface_hub import InferenceClient
import traceback

from django.conf import settings
from ..models import StaffProfile
from tenacity import retry, stop_after_attempt, wait_exponential
import re
import logging

logging.basicConfig(level=logging.DEBUG)


class AIAssistant:

    STATUS_AVAILABLE = 'available'
    STATUS_UNAVAILABLE = 'unavailable'

    FALLBACK_MODELS = [
        "deepseek-ai/DeepSeek-V3",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "Qwen/Qwen2.5-7B-Instruct",
    ]

    def __init__(self):
        self.api_token = settings.HUGGINGFACE_API_TOKEN
        self.model_name = getattr(settings, 'HUGGINGFACE_MODEL_NAME', self.FALLBACK_MODELS[0])
        self.conversation_history = []
        if self.api_token:
            self.client = InferenceClient(
                api_key=self.api_token,
                provider="together"
            )
        else:
            self.client = None

    def add_to_history(self, message, is_user=True):
        self.conversation_history.append({
            'role': 'user' if is_user else 'assistant',
            'content': message
        })

    def get_availability_status(self, staff):
        if staff.status == self.STATUS_AVAILABLE:
            return "Available"
        elif staff.custom_status:
            return f"Unavailable - {staff.custom_status}"
        return "Unavailable"

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=4, max=20))
    def _make_api_request(self, prompt):
        messages = [{"role": "user", "content": prompt}]

        for model in self.FALLBACK_MODELS:
            try:
                logging.info(f"Trying model: {model}")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        self.client.chat_completion,
                        messages=messages,
                        model=model,
                        max_tokens=300,
                        temperature=0.3,
                    )
                    result = future.result(timeout=30)
                self.model_name = model
                return result.choices[0].message.content
            except Exception as e:
                logging.error(f"Model {model} failed: {str(e)}")
                continue

        raise Exception("All models failed. Please try again later.")

    def clean_response(self, text):
        staff_link_pattern = r'<a href="/staff/(\d+)" class="staff-link">([^<]+)</a>'

        def get_correct_name(staff_id):
            try:
                staff = StaffProfile.objects.get(id=staff_id)
                return staff.name
            except StaffProfile.DoesNotExist:
                return '[Unknown Staff]'

        def replace_with_correct_name(match):
            staff_id = match.group(1)
            correct_name = get_correct_name(staff_id)
            return f'<a href="/staff/{staff_id}" class="staff-link">{correct_name}</a>'

        text = re.sub(staff_link_pattern, replace_with_correct_name, text)

        explanation_patterns = [
            r'Step-by-step explanation:.*?(?=The most qualified person|Sorry, from my observation|$)',
            r'Understanding the Query:.*?(?=The most qualified person|Sorry, from my observation|$)',
            r'Here\'s why:.*?(?=The most qualified person|Sorry, from my observation|$)',
            r'Analysis:.*?(?=The most qualified person|Sorry, from my observation|$)',
            r'Let me explain:.*?(?=The most qualified person|Sorry, from my observation|$)',
            r'\d+\.\s+.*?(?=The most qualified person|Sorry, from my observation|$)',
            r'\*\*.*?\*\*',
        ] + [
            f'{starter}.*?(?=The most qualified person|Sorry, from my observation|$)'
            for starter in [
                '<think>', 'Thinking:', 'Let me analyze', 'Let me see',
                'Let me check', 'Let me look', 'Let me find', 'Let me help',
                'First I need to', 'First, I need to', 'I need to',
                'I will first', 'I will check', 'I will look', 'I will search',
                'I will find', 'To answer this', 'Let\'s look at', 'Let\'s see',
            ]
        ]

        for pattern in explanation_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

        staff_links = []
        placeholder_pattern = r'a href="/staff/\d+" class="staff-link"[^/]+/a'
        matches = re.finditer(placeholder_pattern, text)
        for i, match in enumerate(matches):
            staff_links.append(match.group(0))
            text = text.replace(match.group(0), f'STAFFLINK_{i}_PLACEHOLDER')

        text = re.sub(r'(Question:|Answer:|Human:|Assistant:|</think>|First,|Initially,|Finally,|In conclusion,|Therefore,|So,|As a result,)', '', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        for i, link in enumerate(staff_links):
            text = text.replace(f'STAFFLINK_{i}_PLACEHOLDER', link)

        if "The most qualified person" in text:
            parts = text.split("The most qualified person")
            if len(parts) > 2:
                text = "The most qualified person" + parts[1]
            if not text.strip().startswith("The most qualified person"):
                text = "The most qualified person" + text.split("The most qualified person")[1]
        elif "Sorry, from my observation" in text:
            if not text.strip().startswith("Sorry, from my observation"):
                text = "Sorry, from my observation" + text.split("Sorry, from my observation")[1]

        return text.strip()

    def generate_prompt(self, user_query, staff_info, context_text=None, is_email_request=False):
        if is_email_request:
            return f"""<s>[INST] Recent conversation context:
{context_text}

Write a professional email:
- Concise and friendly.
- Avoid mentioning IDs or technical details.
- Use the recipient's name in the greeting.

Format as:
Subject: [Brief subject]

Dear [Staff Member],

[Email content]

Best regards,
[Sender]

[/INST]"""
        else:
            return f"""<s>[INST] Here is our staff directory:

{staff_info}

STRICT RESPONSE FORMAT REQUIREMENTS:

- ALWAYS MENTION THE BEST MATCHED STAFF REGARDLESS OF THEIR AVAILABILITY STATUS
- ONLY NAME AN ALTERNATIVE IF THEY ARE AVAILABLE, AND THEIR ROLES OR SKILLS ARE RELATED TO THE USERS QUERY
- MUST NOT MENTION ANY MATCHED ALTERNATIVE STAFF MEMBERS IF THEY ARE UNAVAILABLE
- THE MENTION OF BEST MATCHED STAFF IS NOT DEPENDANT ON AVAILABILITY
- USE THE EXACT HTML FORMAT PROVIDED BELOW FOR STAFF LINKS
- IN YOUR RESPONSE DO NOT INCLUDE YOUR THOUGHT PROCESS
- KEEP YOUR RESPONSE CONCISE
- USERS MAY MAKE TYPOS SO TRY TO NORMALISE THE TEXT OF THE USER QUERY AS MUCH AS POSSIBLE

Important matching guidelines:
- Use your knowledge to understand relationships between similar skills and terms
- Look for both exact matches and semantically related skills in staff profiles
- Consider the broader context of roles and how they relate to the requested expertise
- ONLY mention an alternative if they are available and their skills or roles are related to the user query
- The best match is the staff member whose skills and roles are most relevant to the user query
- Be consistent with your matching. Different phrasing of the same query should result in the same staff member being mentioned.

Format your concise responses using these exact patterns:

1. For staff mentions, use: <a href="/staff/{{staff_id:NUMBER}}" class="staff-link">[Name]</a>

2. If best match is unavailable (mention them, and also mention alternative if available):
"The most qualified person for this request is <a href="/staff/{{staff_id:NUMBER}}" class="staff-link">[Name]</a> ([Role]) because [reason]. Their current status is: [Status]. However, since they are unavailable, <a href="/staff/{{staff_id:NUMBER}}" class="staff-link">[Name]</a> ([Role]) can help because [reason]. Their status is: [Status]."

3. If best match is available (and no available alternatives):
"The most qualified person for this request is <a href="/staff/{{staff_id:NUMBER}}" class="staff-link">[Name]</a> ([Role]) because [reason]. Their current status is: [Status]."

4. When no one matches the query at all:
"Sorry, from my observation, I do not see anyone in the database that can help you with your query, please look for external help."

Question: {user_query} [/INST]"""

    def get_response(self, user_query):
        if not self.client:
            return "AI assistant is currently unavailable. Please contact the administrator to set up the Hugging Face API token."

        try:
            self.conversation_history = []

            all_staff = StaffProfile.objects.all()
            staff_info = "\n".join([
                f"Staff Member: {staff.name}"
                f"\nPrimary Role: {staff.role}"
                f"\nRole Description: {staff.bio or 'Not specified'}"
                f"\nDepartment: {staff.department}"
                f"\nCore Skills: {staff.skills or 'Not specified'}"
                f"\nAbout: {staff.about_me or 'Not specified'}"
                f"\nStatus: {self.get_availability_status(staff)}"
                f"\nEmail: {staff.email}"
                f"\nID: {staff.id}\n"
                for staff in all_staff
            ])

            email_phrases = [
                'write an email', 'compose an email', 'make email',
                'create an email', 'draft an email', 'send an email', 'write email'
            ]
            is_email_request = any(phrase in user_query.lower() for phrase in email_phrases)

            logging.debug(f"User Query: {user_query}, Detected Email Request: {is_email_request}")

            recent_context = None
            prompt = self.generate_prompt(user_query, staff_info, recent_context, is_email_request)

            try:
                generated_text = self._make_api_request(prompt)
                cleaned_response = self.clean_response(generated_text)

                if is_email_request:
                    self.add_to_history(user_query, is_user=True)
                    self.add_to_history(cleaned_response, is_user=False)

                return cleaned_response

            except Exception as api_error:
                traceback.print_exc()
                logging.error(f"API Error: {str(api_error)}")
                return "I'm having trouble with the AI service. Please try again in a few moments."

        except Exception as e:
            traceback.print_exc()
            logging.error(f"Error in get_response: {str(e)}")
            return "I'm having trouble processing your request. Please try again."
