"""The input box: a line between two rules, with the command list open beneath it."""
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.styles import Style

VISIBLE_ITEMS = 8
STYLE = Style.from_dict({"rule": "#6f6a60", "prompt": "#d97757 bold", "item": "#efeadf", "meta": "#9c978c",
                         "selected": "#d97757 bold", "status": "#9c978c"})


class InputBox:
    def __init__(self, completer, hold: set[str]):
        """hold names commands that need an argument, so Enter waits for it instead of running them."""
        self.completer, self.hold = completer, hold
        self.items, self.selected, self.browsing, self.status = [], None, False, ""
        self.past, self.position, self.draft = [], 0, ""
        self.buffer = Buffer(multiline=False, on_text_changed=self.changed)
        row = VSplit([Window(FormattedTextControl([("class:prompt", "> ")]), width=2, height=1),
                      Window(BufferControl(buffer=self.buffer), wrap_lines=True, dont_extend_height=True)])
        rule = lambda: Window(char="─", height=1, style="class:rule")
        footer = Window(FormattedTextControl(self.footer), dont_extend_height=True)
        self.app = Application(layout=Layout(HSplit([rule(), row, rule(), footer]), focused_element=self.buffer),
                               key_bindings=self.bindings(), style=STYLE, full_screen=False, erase_when_done=True)

    def read(self, status: str) -> str:
        self.status = status
        self.changed()
        return self.app.run()

    def naming_a_command(self) -> bool:
        text = self.buffer.text.lstrip()
        return text.startswith("/") and " " not in text

    def changed(self, _=None) -> None:
        """Refresh the list; recalled history is not something to autocomplete."""
        self.items = [] if self.browsing else list(self.completer.get_completions(self.buffer.document, None))
        self.selected = 0 if self.items and self.naming_a_command() else None

    def apply(self, item) -> None:
        position = self.buffer.cursor_position + item.start_position
        text = self.buffer.text[:position] + item.text
        self.buffer.document = Document(text, len(text))

    def recall(self, step: int) -> None:
        if self.position == len(self.past):
            self.draft = self.buffer.text
        self.position = max(0, min(self.position + step, len(self.past)))
        text = [*self.past, self.draft][self.position]
        self.browsing = True
        self.buffer.document = Document(text, len(text))
        self.browsing = False

    def bindings(self) -> KeyBindings:
        keys = KeyBindings()

        @keys.add("enter")
        def submit(event):
            if self.selected is not None:
                naming = self.naming_a_command()
                self.apply(self.items[self.selected])
                if naming and self.buffer.text.strip() in self.hold:
                    self.buffer.insert_text(" ")
                    return
            line = self.buffer.text
            if line.strip() and line not in self.past[-1:]:
                self.past.append(line)
            self.position, self.draft = len(self.past), ""
            self.buffer.reset()
            event.app.exit(result=line)

        @keys.add("tab")
        def fill(event):
            if self.items:
                naming = self.naming_a_command()
                self.apply(self.items[self.selected or 0])
                if naming:
                    self.buffer.insert_text(" ")

        def move(step):
            if self.items:
                self.selected = ((self.selected if self.selected is not None else -step) + step) % len(self.items)
            else:
                self.recall(step)

        @keys.add("up")
        def previous(event):
            move(-1)

        @keys.add("down")
        def following(event):
            move(1)

        @keys.add("c-c")
        def clear(event):
            self.buffer.reset()

        @keys.add("c-d")
        def leave(event):
            if not self.buffer.text:
                event.app.exit(exception=EOFError())

        return keys

    def footer(self):
        if not self.items:
            return [("class:status", self.status)]
        focus = self.selected or 0
        first = max(0, min(focus - VISIBLE_ITEMS + 1, len(self.items) - VISIBLE_ITEMS))
        window = self.items[first:first + VISIBLE_ITEMS]
        width = max(len(item.display_text) for item in window) + 4
        lines = []
        for offset, item in enumerate(window, first):
            chosen = offset == self.selected
            lines += [("class:selected" if chosen else "class:item", f"  {item.display_text:<{width}}"),
                      ("class:selected" if chosen else "class:meta", item.display_meta_text), ("", "\n")]
        return lines[:-1]
