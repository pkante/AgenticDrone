import re
from enum import Enum
from typing import Optional, List, Union
from .abs.skill_item import SkillItem, SkillArg

class SkillSetLevel(Enum):
    LOW = "low"
    HIGH = "high"

class SkillSet():
    def __init__(self, level = "low", lower_level_skillset: 'SkillSet' = None):
        self.skills = {}
        self.level = SkillSetLevel(level)
        self.lower_level_skillset = lower_level_skillset
    
    def get_skill(self, skill_name: str) -> Optional[SkillItem]:
        """Returns a SkillItem by its name or abbr."""
        skill = None
        if skill_name in self.skills:
            skill = self.skills[skill_name]
        elif skill_name in SkillItem.abbr_dict:
            skill = self.skills.get(SkillItem.abbr_dict[skill_name])
        return skill
    
    def add_skill(self, skill_item: SkillItem):
        """Adds a SkillItem to the set."""
        if skill_item.skill_name in self.skills:
            raise ValueError(f"A skill with the name '{skill_item.skill_name}' already exists.")
        # Set the low-level skillset for high-level skills
        if self.level == SkillSetLevel.HIGH and isinstance(skill_item, HighLevelSkillItem):
            if self.lower_level_skillset is not None:
                skill_item.set_skillset(self.lower_level_skillset, self)
            else:
                raise ValueError("Low-level skillset is not set.")

        self.skills[skill_item.skill_name] = skill_item
    
    def remove_skill(self, skill_name: str):
        """Removes a SkillItem from the set by its name."""
        if skill_name not in self.skills:
            raise ValueError(f"No skill found with the name '{skill_name}'.")
        # remove skill by value
        del self.skills[skill_name]
    
    def __repr__(self) -> str:
        string = ""
        for skill in self.skills.values():
            string += f"{skill}\n"
        return string

class LowLevelSkillItem(SkillItem):
    def __init__(self, skill_name: str, skill_callable: callable,
                 skill_description: str = "", args: List[SkillArg] = []):
        self.skill_name = skill_name
        self.abbr = self.generate_abbreviation(skill_name)
        self.abbr_dict[self.abbr] = skill_name
        self.skill_callable = skill_callable
        self.skill_description = skill_description
        self.args = args

    def get_name(self) -> str:
        return self.skill_name
    
    def get_skill_description(self) -> str:
        return self.skill_description
    
    def get_argument(self) -> List[SkillArg]:
        return self.args
    
    def execute(self, arg_list: List[Union[int, float, str]]):
        """Executes the skill with the provided arguments."""
        if callable(self.skill_callable):
            parsed_args = self.parse_args(arg_list)
            return self.skill_callable(*parsed_args)
        else:
            raise ValueError(f"'{self.skill_callable}' is not a callable function.")

    def __repr__(self) -> str:
        return (f"abbr:{self.abbr},"
                f"name:{self.skill_name},"
                f"args:{[arg for arg in self.args]},"
                f"description:{self.skill_description}")

class HighLevelSkillItem(SkillItem):
    def __init__(self, skill_name: str, definition: str,
                 skill_description: str = ""):
        self.skill_name = skill_name
        self.abbr = self.generate_abbreviation(skill_name)
        self.abbr_dict[self.abbr] = skill_name
        self.definition = definition
        self.skill_description = skill_description
        self.low_level_skillset = None
        self.args = []

    def load_from_dict(skill_dict: dict):
        return HighLevelSkillItem(skill_dict["skill_name"], skill_dict["definition"], skill_dict["skill_description"])

    def get_name(self) -> str:
        return self.skill_name
    
    def get_skill_description(self) -> str:
        return self.skill_description
    
    def get_argument(self) -> List[SkillArg]:
        return self.args

    def set_skillset(self, low_level_skillset: SkillSet, high_level_skillset: SkillSet):
        self.low_level_skillset = low_level_skillset
        self.high_level_skillset = high_level_skillset
        self.args = self.generate_argument_list()

    def generate_argument_list(self) -> List[SkillArg]:
        # Extract all skill calls with their arguments from the code
        skill_calls = re.findall(r'(\w+)\(([^)]*)\)', self.definition)

        arg_types = {}

        for skill_name, args in skill_calls:
            args = [a.strip() for a in args.split(',')]
            if skill_name == "int":
                function_args = [SkillArg("value", int)]
            elif skill_name == "float":
                function_args = [SkillArg("value", float)]
            elif skill_name == "str":
                function_args = [SkillArg("value", str)]
            else:
                skill = self.low_level_skillset.get_skill(skill_name)
                if skill is None:
                    skill = self.high_level_skillset.get_skill(skill_name)

                if skill is None:
                    raise ValueError(f"Skill '{skill_name}' not found in the low-level or high-level skillset.")

                function_args = skill.get_argument()
            for i, arg in enumerate(args):
                if arg.startswith('$') and arg not in arg_types:
                    # Match the positional argument with its type from the function definition
                    arg_types[arg] = function_args[i]

        # Convert the mapped arguments to a user-friendly list in order of $position
        arg_types = dict(sorted(arg_types.items()))
        arg_list = [arg for arg in arg_types.values()]

        return arg_list

    def execute(self, arg_list: List[Union[int, float, str]]):
        """Executes the skill with the provided arguments."""
        if self.low_level_skillset is None:
            raise ValueError("Low-level skillset is not set.")
        if len(arg_list) != len(self.args):
            raise ValueError(f"Expected {len(self.args)} arguments, but got {len(arg_list)}.")
        # replace all $1, $2, ... with segments
        definition = self.definition
        for i in range(0, len(arg_list)):
            definition = definition.replace(f"${i + 1}", arg_list[i])
        return definition

    def __repr__(self) -> str:
        return (f"abbr:{self.abbr},"
                f"name:{self.skill_name},"
                f"definition:{self.definition},"
                f"args:{[arg for arg in self.args]},"
                f"description:{self.skill_description}")