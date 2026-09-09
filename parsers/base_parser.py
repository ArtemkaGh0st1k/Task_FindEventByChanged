from abc import ABC, abstractmethod


class BaseParser(ABC):

    @abstractmethod
    def extract_json_from_text(self, raw_text: str) -> dict:
        pass