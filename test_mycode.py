from unittest.mock import MagicMock

import pytest
from box import Box
from langchain_core.messages import HumanMessage
from langchain_core.prompt_values import ChatPromptValue

from mycode import tellJoke


class MockStrOutputParser:
    def __call__(self, input_data):
        return input_data.__rshift__.return_value[0].text


def test_tellJoke(mocker):
    # Mock the AzureChatOpenAI class
    mock_azure_chat_openai = MagicMock()
    mocker.patch("mycode.AzureChatOpenAI", return_value=mock_azure_chat_openai)

    # Configure the mock instance
    mock_instance = mock_azure_chat_openai.return_value
    mock_instance.__rshift__.return_value = [Box({"text": "Mocked joke"})]

    # Mock the StrOutputParser class
    mock_str_output_parser = MockStrOutputParser()

    # Replace the StrOutputParser instance in the chain with the mock
    mocker.patch("mycode.StrOutputParser", return_value=mock_str_output_parser)

    # Call the function under test
    joke = tellJoke("bears")

    # Assert the result
    assert joke == "Mocked joke"

    print(mock_azure_chat_openai.call_args)

    # Optional: Assert that the AzureChatOpenAI class was called
    mock_azure_chat_openai.assert_called_once()

    # Optional: Assert that the AzureChatOpenAI class was called with the correct arguments
    mock_azure_chat_openai.assert_called_once_with(
        ChatPromptValue(
            messages=[
                HumanMessage(
                    content="tell me a joke about bears",
                    additional_kwargs={},
                    response_metadata={},
                )
            ]
        )
    )


if __name__ == "__main__":
    pytest.main()
