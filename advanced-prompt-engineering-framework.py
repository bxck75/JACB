import abc
from typing import Dict, List, Any, Callable
import re

class PromptComponent(abc.ABC):
    @abc.abstractmethod
    def process(self, context: Dict[str, Any]) -> str:
        pass

class TextComponent(PromptComponent):
    def __init__(self, text: str):
        self.text = text
    
    def process(self, context: Dict[str, Any]) -> str:
        return self.text

class VariableComponent(PromptComponent):
    def __init__(self, variable_name: str):
        self.variable_name = variable_name
    
    def process(self, context: Dict[str, Any]) -> str:
        return str(context.get(self.variable_name, f"<{self.variable_name}>"))

class ConditionalComponent(PromptComponent):
    def __init__(self, condition: Callable[[Dict[str, Any]], bool], true_component: PromptComponent, false_component: PromptComponent):
        self.condition = condition
        self.true_component = true_component
        self.false_component = false_component
    
    def process(self, context: Dict[str, Any]) -> str:
        if self.condition(context):
            return self.true_component.process(context)
        else:
            return self.false_component.process(context)

class LoopComponent(PromptComponent):
    def __init__(self, iterable_name: str, item_name: str, loop_component: PromptComponent):
        self.iterable_name = iterable_name
        self.item_name = item_name
        self.loop_component = loop_component
    
    def process(self, context: Dict[str, Any]) -> str:
        result = []
        for item in context.get(self.iterable_name, []):
            item_context = context.copy()
            item_context[self.item_name] = item
            result.append(self.loop_component.process(item_context))
        return "".join(result)

class DynamicPrompt(PromptComponent):
    def __init__(self, components: List[PromptComponent]):
        self.components = components
    
    def process(self, context: Dict[str, Any]) -> str:
        return "".join(component.process(context) for component in self.components)

class PromptTemplate:
    def __init__(self, template: str):
        self.components = self._parse_template(template)
    
    def _parse_template(self, template: str) -> List[PromptComponent]:
        components = []
        parts = re.split(r'(\{\{.*?\}\})', template)
        for part in parts:
            if part.startswith('{{') and part.endswith('}}'):
                variable = part[2:-2].strip()
                components.append(VariableComponent(variable))
            else:
                components.append(TextComponent(part))
        return components
    
    def process(self, context: Dict[str, Any]) -> str:
        return DynamicPrompt(self.components).process(context)

class PromptLibrary:
    def __init__(self):
        self.prompts: Dict[str, PromptComponent] = {}
    
    def add_prompt(self, name: str, prompt: PromptComponent):
        self.prompts[name] = prompt
    
    def get_prompt(self, name: str) -> PromptComponent:
        return self.prompts.get(name)

class PromptOrchestrator:
    def __init__(self, library: PromptLibrary):
        self.library = library
        self.context: Dict[str, Any] = {}
    
    def set_context(self, key: str, value: Any):
        self.context[key] = value
    
    def generate_prompt(self, prompt_name: str) -> str:
        prompt = self.library.get_prompt(prompt_name)
        if prompt:
            return prompt.process(self.context)
        else:
            raise ValueError(f"Prompt '{prompt_name}' not found in library")

# Example usage
library = PromptLibrary()

# Define a complex prompt with conditional and loop components
story_prompt = DynamicPrompt([
    TextComponent("Once upon a time, there was a "),
    VariableComponent("protagonist"),
    TextComponent(" who lived in a "),
    VariableComponent("setting"),
    TextComponent(". "),
    ConditionalComponent(
        lambda ctx: ctx.get("has_quest", False),
        TextComponent("They embarked on a great quest. "),
        TextComponent("They lived a peaceful life. ")
    ),
    LoopComponent(
        "challenges",
        "challenge",
        DynamicPrompt([
            TextComponent("They faced the challenge of "),
            VariableComponent("challenge"),
            TextComponent(". ")
        ])
    ),
    TextComponent("And they lived "),
    ConditionalComponent(
        lambda ctx: ctx.get("happy_ending", True),
        TextComponent("happily ever after."),
        TextComponent("with many more adventures to come.")
    )
])

library.add_prompt("story", story_prompt)

orchestrator = PromptOrchestrator(library)
orchestrator.set_context("protagonist", "brave knight")
orchestrator.set_context("setting", "magical forest")
orchestrator.set_context("has_quest", True)
orchestrator.set_context("challenges", ["a fierce dragon", "a cunning wizard"])
orchestrator.set_context("happy_ending", False)

generated_prompt = orchestrator.generate_prompt("story")
print(generated_prompt)