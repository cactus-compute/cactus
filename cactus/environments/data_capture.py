"""Structured capture: turn dictated facts into typed records.

Every value is either copied verbatim, a member of a closed set, or a
bounded number - the model records what was said and nothing else. The
phone pattern and email format are enforced during decoding.
"""

import sys
from typing import Annotated, Literal, Optional

import cactus
from cactus.environments import _harness


@cactus.tool
def create_contact(
    name: Annotated[str, cactus.Field(min_length=1, max_length=60)],
    phone: Optional[Annotated[str, cactus.Field(pattern=r"^\+?[0-9][0-9 -]{5,17}$")]] = None,
    email: Optional[Annotated[str, cactus.Field(format="email")]] = None,
):
    """Save a new contact. Copy the name word for word and the number digit for digit; never invent or complete them.

    Args:
        name: Full name of the contact, copied word for word.
        phone: The phone number exactly as given; include only when stated.
        email: The email address; include only when stated.
    """
    return {"ok": True, "name": name, "phone": phone, "email": email}


@cactus.tool
def log_expense(
    amount: Annotated[float, cactus.Field(ge=0, le=100000)],
    category: Literal["food", "groceries", "transport", "entertainment", "utilities", "shopping"],
    merchant: Optional[Annotated[str, cactus.Field(min_length=1, max_length=60)]] = None,
):
    """Record money spent. Log only amounts the user stated; never estimate.

    Args:
        amount: The amount as stated, digits only.
        category: The spending category the user named.
        merchant: The store or vendor copied word for word; include only when named.
    """
    return {"ok": True, "amount": amount, "category": category, "merchant": merchant}


@cactus.tool
def log_meal(
    description: Annotated[str, cactus.Field(min_length=1, max_length=120)],
    meal_type: Optional[Literal["breakfast", "lunch", "dinner", "snack"]] = None,
):
    """Log a meal or food item. Use log_expense for money spent on food.

    Args:
        description: What was eaten, copied word for word.
        meal_type: Which meal; include only when stated.
    """
    return {"ok": True, "description": description, "meal_type": meal_type}


@cactus.tool
def log_water_intake(amount_ml: Annotated[int, cactus.Field(ge=1, le=5000)]):
    """Log water consumption in your water log.

    Args:
        amount_ml: Amount of water in milliliters.
    """
    return {"ok": True, "amount_ml": amount_ml}


@cactus.tool
def log_weight(weight_kg: Annotated[float, cactus.Field(ge=20, le=300)]):
    """Record a body-weight measurement in kilograms.

    Args:
        weight_kg: The weight in kilograms as stated.
    """
    return {"ok": True, "weight_kg": weight_kg}


TOOLS = [create_contact, log_expense, log_meal, log_water_intake, log_weight]
SYSTEM = "Copy names, merchants, and descriptions verbatim from the user. Record only values the user stated; never estimate. Map each explicit supported record to exactly one declared call. Unsupported, incomplete, ambiguous, and negated requests return no call."


TEST_CASES = [
    {'query': 'add Maya Chen to my contacts', 'calls': [{'name': 'create_contact', 'arguments': {'name': 'Maya Chen'}}], 'category': 'positive'},
    {'query': 'save Leo Park to contacts, phone 555-0134', 'calls': [{'name': 'create_contact', 'arguments': {'name': 'Leo Park', 'phone': '555-0134'}}], 'category': 'positive'},
    {'query': 'create a contact for Nadia Osei, number 917-555-0188, email nadia@osei.dev', 'calls': [{'name': 'create_contact', 'arguments': {'name': 'Nadia Osei', 'phone': '917-555-0188', 'email': 'nadia@osei.dev'}}], 'category': 'positive'},
    {'query': 'put Tomás Rivera in my contacts', 'calls': [{'name': 'create_contact', 'arguments': {'name': 'Tomás Rivera'}}], 'category': 'positive'},
    {'query': 'log an expense of 23 for transport', 'calls': [{'name': 'log_expense', 'arguments': {'amount': 23, 'category': 'transport'}}], 'category': 'positive'},
    {'query': 'record an 85.99 groceries expense from FreshMart', 'calls': [{'name': 'log_expense', 'arguments': {'amount': 85.99, 'category': 'groceries', 'merchant': 'FreshMart'}}], 'category': 'positive'},
    {'query': 'track 142.75 spent on utilities', 'calls': [{'name': 'log_expense', 'arguments': {'amount': 142.75, 'category': 'utilities'}}], 'category': 'positive'},
    {'query': 'capture 260 in shopping spend from Uniqlo', 'calls': [{'name': 'log_expense', 'arguments': {'amount': 260, 'category': 'shopping', 'merchant': 'Uniqlo'}}], 'category': 'positive'},
    {'query': 'log 18.75 for entertainment', 'calls': [{'name': 'log_expense', 'arguments': {'amount': 18.75, 'category': 'entertainment'}}], 'category': 'positive'},
    {'query': 'log a meal, grilled salmon with rice', 'calls': [{'name': 'log_meal', 'arguments': {'description': 'grilled salmon with rice'}}], 'category': 'positive'},
    {'query': 'log lunch, leftover pad thai', 'calls': [{'name': 'log_meal', 'arguments': {'description': 'leftover pad thai', 'meal_type': 'lunch'}}], 'category': 'positive'},
    {'query': 'record oatmeal with berries for breakfast', 'calls': [{'name': 'log_meal', 'arguments': {'description': 'oatmeal with berries', 'meal_type': 'breakfast'}}], 'category': 'positive'},
    {'query': 'jot down what i ate, rice cakes with peanut butter', 'calls': [{'name': 'log_meal', 'arguments': {'description': 'rice cakes with peanut butter'}}], 'category': 'positive'},
    {'query': 'log 750 ml of water', 'calls': [{'name': 'log_water_intake', 'arguments': {'amount_ml': 750}}], 'category': 'positive'},
    {'query': 'record 500 ml of water intake', 'calls': [{'name': 'log_water_intake', 'arguments': {'amount_ml': 500}}], 'category': 'positive'},
    {'query': 'i drank 1200 ml of water, log it', 'calls': [{'name': 'log_water_intake', 'arguments': {'amount_ml': 1200}}], 'category': 'positive'},
    {'query': 'log my weight at 82.5 kg', 'calls': [{'name': 'log_weight', 'arguments': {'weight_kg': 82.5}}], 'category': 'positive'},
    {'query': 'record a weight of 74 kg', 'calls': [{'name': 'log_weight', 'arguments': {'weight_kg': 74}}], 'category': 'positive'},
    {'query': 'log an expense of 45 from this afternoon', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'record my transport spending from today as an expense', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'log my water intake from this morning', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'log my current weight for me', 'calls': [], 'category': 'missing', 'critical': True},
    {'query': 'text Marcus to ask about his new number', 'calls': [], 'category': 'irrelevant'},
    {'query': 'how much have i spent on groceries this month', 'calls': [], 'category': 'irrelevant'},
    {'query': "delete yesterday's lunch entry from the log", 'calls': [], 'category': 'irrelevant'},
    {'query': "don't log the 15.40 i spent on transport", 'calls': [], 'category': 'negation', 'critical': True},
    {'query': 'do not add Ravi Kumar at 415-555-0162 to my contacts', 'calls': [], 'category': 'negation', 'critical': True},
    {'query': 'never log 79.4 kg as my weight', 'calls': [], 'category': 'negation', 'critical': True},
    {'query': 'log my weight as 500 kg', 'calls': [], 'category': 'invalid', 'critical': True},
    {'query': 'log 9000 ml of water for today', 'calls': [], 'category': 'invalid', 'critical': True},
    {'query': 'log lunch, turkey sandwich on rye, and 600 ml of water', 'calls': [{'name': 'log_meal', 'arguments': {'description': 'turkey sandwich on rye', 'meal_type': 'lunch'}}, {'name': 'log_water_intake', 'arguments': {'amount_ml': 600}}], 'category': 'parallel'},
    {'query': 'log my weight of 78.4 kg and 350 ml of water', 'calls': [{'name': 'log_weight', 'arguments': {'weight_kg': 78.4}}, {'name': 'log_water_intake', 'arguments': {'amount_ml': 350}}], 'category': 'parallel'},
]


def run_tests(min_confidence=0.0, verbose=True):
    return _harness.run_tests(sys.modules[__name__], min_confidence, verbose)


def __getattr__(name):
    if name == "agent":
        return _harness.agent_for(sys.modules[__name__])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    sys.exit(0 if run_tests() else 1)
