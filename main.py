from fileinput import filename
from nicegui import ui
from datetime import datetime
import asyncio
import json
import os
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
import concurrent.futures
import sys

# ===== Lazy LLM =====
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini")

llm = get_llm()
os.makedirs("stretch-guide-output", exist_ok=True)

# ===== Configuration JSON =====
CONFIG_JSON = """
{
  "title": "Stretch Guide Generator",
  "subtitle": "Creates stretch guides with instructions, benefits, and precautions for improving flexibility and prevent injuries. Pick an area → muscles auto-populate → choose a muscle → stretches auto-populate → generate guide.",
  "sections": [
    {
      "title": "PRIMARY GUEST",
      "layout": "two_columns",
      "fields": [
        {"id": "first_name", "label": "First Name", "type": "text", "width": "w-1/4"},
        {"id": "last_name", "label": "Last Name", "type": "text", "width": "w-1/4"}
      ]
    },
    {
      "title": "PREFERENCES",
      "layout": "three_cards",
      "cards": [
        {
          "title": "Area",
          "fields": [
            {"id": "arms", "label": "Arms", "type": "radio"},
            {"id": "legs", "label": "Legs", "type": "radio"},
            {"id": "chest", "label": "Chest", "type": "radio"},
            {"id": "abs", "label": "Abs", "type": "radio"},
            {"id": "back", "label": "Back", "type": "radio"},
            {"id": "shoulders", "label": "Shoulders", "type": "radio"},
            {"id": "neck", "label": "Neck", "type": "radio"}
          ]
        },
        {
          "title": "Muscle",
          "fields": []
        },
        {
          "title": "Stretches",
          "fields": []
        }
      ]
    }
  ],
  "primary_action": {"label": "Generate Guide"}
}
"""
# ===== Shared result state =====
result_data = {"title": "", "content": ""}


# ===== resilient LLM caller =====
async def call_llm_async(prompt: str) -> str:
    # try common async names
    for name in ("ainvoke", "arun", "agenerate", "apredict"):
        fn = getattr(llm, name, None)
        if callable(fn):
            try:
                res = await fn(prompt)
                if isinstance(res, str):
                    return res
                if hasattr(res, "content"):
                    return getattr(res, "content")
                if hasattr(res, "generations"):
                    gens = getattr(res, "generations")
                    if isinstance(gens, list) and gens:
                        first = gens[0]
                        if isinstance(first, list) and first and hasattr(first[0], "text"):
                            return first[0].text
                        if hasattr(first, "text"):
                            return first.text
                return str(res)
            except Exception:
                continue

    # fallback to sync methods run in thread
    for name in ("invoke", "run", "generate", "predict"):
        fn = getattr(llm, name, None)
        if callable(fn):
            loop = asyncio.get_running_loop()
            try:
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    res = await loop.run_in_executor(pool, lambda: fn(prompt))
                if isinstance(res, str):
                    return res
                if hasattr(res, "content"):
                    return getattr(res, "content")
                return str(res)
            except Exception:
                continue

    raise RuntimeError("No usable LLM method found on 'llm' object.")

# ===== helper types =====
WidgetRef = Dict[str, Any]

# ===== UI builder with dynamic cards implementation =====
def build_ui_from_config(config: Dict[str, Any]):
    state: Dict[str, Any] = {
        "widgets": {},
        "area": None,
        "muscle": None,
        "stretch": None,
    }

    # progress dialog
    with ui.dialog() as progress_dialog, ui.card():
        progress_label = ui.label("⏳ Please wait...")
        ui.spinner(size="lg")

    def start_progress(msg: str):
        progress_label.text = msg
        progress_dialog.open()

    def stop_progress():
        try:
            progress_dialog.close()
        except Exception:
            pass

    def set_status(txt: str):
        if state.get("status_label"):
            state["status_label"].text = txt

    def update_generate_button_state():
        print("🔄 Checking generate button state...")
        btn = state.get("generate_button")
        first = state["widgets"].get("first_name")
        first_val = (first.value or "").strip() if first else ""
        if not btn:
            return

        is_enabled = bool(first_val and state.get("area") and state.get("muscle") and state.get("stretch"))
        if is_enabled:
            btn.enable()
        else:
            btn.disable()


    # find cards list in config: supports sections[...] with layout 'three_cards' or top-level 'cards'
    def find_cards(cfg: Dict[str, Any]):
        # Search sections first
        for sec in cfg.get("sections", []):
            if sec.get("layout") == "three_cards" and sec.get("cards"):
                return sec["cards"]
        # fallback to top-level 'cards'
        if cfg.get("cards"):
            return cfg["cards"]
        return None

    cards = find_cards(config)
    # fallback area options if config doesn't provide them

    # async functions to populate muscle and stretches cards
    async def populate_muscles_for_area(area: str, muscle_card_container):
        if not area:
            return
        state["muscle"] = None
        state["stretch"] = None
        muscle_content.clear()
        stretch_content.clear()
        start_progress(f"Fetching muscles for {area}...")
        try:
            prompt = f"List 5 key muscles in the human {area}. Return as a plain newline-separated list, e.g. 'Pectoralis Major' on each line."
            raw = await call_llm_async(prompt)
            muscles = [line.strip().lstrip("-• ").strip() for line in raw.splitlines() if line.strip()]
            
            with muscle_card_container:
                # ui.label(f"Muscles in {area}").classes("font-bold text-lg")
                if muscles:
                    def on_muscle_change(e):
                        selected = getattr(e, "value", None)
                        state["muscle"] = selected
                        state["stretch"] = None
                        # clear and repopulate stretches
                        stretch_card_container.clear()
                        asyncio.create_task(populate_stretches_for_muscle(selected, stretch_card_container))
                        update_generate_button_state()
                    ui.radio(muscles, on_change=on_muscle_change)
                else:
                    ui.label("_No muscles returned_")
        except Exception as ex:
            show_message("Error", f"Error fetching muscles: {ex}")
        finally:
            stop_progress()

    async def populate_stretches_for_muscle(muscle: str, stretch_card_container):
        start_progress(f"Generating stretches...")
        if not muscle:
            return
        state["stretch"] = None
        stretch_content.clear()
        start_progress(f"Fetching stretches for {muscle}...")
        try:
            prompt = f"List 5 common stretches that specifically target the {muscle}. Return as a plain newline-separated list, e.g. 'Triceps Stretch' on each line."
            raw = await call_llm_async(prompt)
            stretches = [line.strip().lstrip("-• ").strip() for line in raw.splitlines() if line.strip()]
            with stretch_card_container:
                ui.label(f"Stretches for {muscle}").classes("font-bold text-lg")
                if stretches:
                    def on_stretch_change(e):
                        state["stretch"] = getattr(e, "value", None)
                        update_generate_button_state()
                    ui.radio(stretches, on_change=on_stretch_change)
                else:
                    ui.label("_No stretches returned_")


        except Exception as ex:
            await ui.notify(f"Error fetching stretches: {ex}", type="negative")
        finally:
            stop_progress()

    # Generate button action
    async def generate_guide_action():
        # code here...
        first_widget = state["widgets"].get("first_name")
        last_widget = state["widgets"].get("last_name")
        first = (first_widget.value or "").strip() if first_widget else ""
        last = (last_widget.value or "").strip() if last_widget else ""
        area = state.get("area")
        muscle = state.get("muscle")
        stretch = state.get("stretch")

        if not first or not area or not muscle or not stretch:
            await ui.notify("❌ Please complete Name, Area, Muscle, and Stretch.", type="warning")
            return

        # set_status("Generating guide...")
        start_progress("Generating stretch guide...")
        btn = state.get("generate_button")
        try:
            if btn:
                btn.props("loading")
            prompt = f"""
            Create a detailed stretch guide for:
            - Name: {first} {last}
            - Area: {area}
            - Muscle: {muscle}
            - Stretch: {stretch}

            Include: step-by-step instructions, benefits, and precautions.
            Return the guide in Markdown format.
            """
            result = await call_llm_async(prompt)

            # Save to file
            filename = f"stretch-guide-output/stretch_guide_{first}_{last}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result)

            # Update shared state
            result_data["title"] = f"📝 Stretch Guide for {first} {last} ({area} - {muscle} - {stretch} )"
            result_data["content"] = result
            stop_progress()
            ui.navigate.to("/result")
            # Show result in dialog

            with ui.dialog() as dlg, ui.card():
                ui.label("📥 Generated Stretch Guide").classes("text-lg font-bold")
                ui.markdown(result).classes("prose max-w-none")
                ui.button("Close", on_click=dlg.close)
            dlg.open()
            # show_message("✅ Success", f"Guide saved as {filename}")
        except Exception as ex:
            show_message("❌ Error", f"Failed to generate guide: {ex}")
        finally:
            if btn:
                try:
                    btn.props(remove="loading")
                except Exception:
                    pass
            set_status("")
            update_generate_button_state()

    # ----- Build UI -----
    with ui.card().classes("max-w-4xl mx-auto p-6 space-y-4"):
        ui.label(config.get("title", "Generator")).classes("text-2xl font-bold text-center")
        if config.get("subtitle"):
            ui.label(config["subtitle"]).classes("text-sm text-center text-gray-500 mb-4")

                # Render primary guest section fields (if any)
        for sec in config.get("sections", []):
            title = sec.get("title")
            if title:
                ui.label(title).classes("text-md font-semibold mt-2")

            layout = sec.get("layout")
            fields = sec.get("fields", [])

            if layout == "two_columns":
                with ui.row().classes("w-full gap-4"):
                    for f in fields:
                        t = f.get("type", "text")
                        width = f.get("width", "w-1/2")
                        label = f.get("label", f.get("id"))
                        if t in ("text", "input"):
                            w = ui.input(label).classes(width)
                            state["widgets"][f["id"]] = w
                        else:
                            w = ui.input(label).classes(width)
                            state["widgets"][f["id"]] = w

            elif layout == "row":
                with ui.row().classes("w-full gap-4"):
                    for f in fields:
                        t = f.get("type", "text")
                        label = f.get("label", f.get("id"))
                        if t in ("text", "input"):
                            w = ui.input(label).classes("w-1/2")
                            state["widgets"][f["id"]] = w
                        else:
                            w = ui.input(label)
                            state["widgets"][f["id"]] = w

        # status label
        state["status_label"] = ui.label("").classes("text-sm text-gray-600")

        # Three card area (cards variable)
        with ui.row().classes("w-full gap-6"):
            # Determine area options: prefer config["area"], then cards' labels, then defaults
            area_from_config = config.get("area")
            if isinstance(area_from_config, list) and area_from_config:
                area_options: List[str] = [str(x) for x in area_from_config]
            if cards and isinstance(cards, list) and len(cards) >= 1:
                area_card = cards[0]
                # if the area card lists individual radio fields, use their labels
                field_labels: List[str] = []
                for fld in area_card.get("fields", []):
                    lbl = fld.get("label") or fld.get("id")
                    if lbl:
                        field_labels.append(lbl)
                if field_labels:
                    area_options = field_labels
            else:
                area_card = {"title": "Area", "fields": []}
            

            # Wrap all 3 cards inside a row so they sit side by side
            with ui.row().classes("w-full grid grid-cols-1 md:grid-cols-3 gap-4 items-start"):
                
                # Card 1 - Area
                with ui.card().classes("flex-1 min-w-[250px] max-w-[300px] p-4 h-full"):
                    ui.label(area_card.get("title", "Area")).classes("text-lg font-semibold")

                    async def on_area_change(e):
                        selected = getattr(e, "value", None)
                        state["area"] = selected
                        update_generate_button_state()
                        await populate_muscles_for_area(selected, muscle_card_container)

                    state["area_radio_widget"] = ui.radio(area_options, on_change=on_area_change)
                
                # Card 2 - Muscle (initially empty; will populate)
                muscle_card_container = ui.card().classes("flex-1 min-w-[250px] max-w-[300px] p-4 h-full")
                with muscle_card_container:
                    ui.label("Muscles").classes("text-lg font-semibold")
                    muscle_content = ui.column()  # inner container you can clear later
                    # ui.label("_Select an area first_", parent=muscle_content)
                
                # Card 3 - Stretches (initially empty; will populate)
                stretch_card_container = ui.card().classes("flex-1 min-w-[250px] max-w-[300px] p-4 h-full")
                with stretch_card_container:
                    ui.label("Stretches").classes("text-lg font-semibold")
                    stretch_content = ui.column()  # inner container you can clear later
                    # ui.label("_Select a muscle first_", parent=stretch_content)



        # primary action button
        state["generate_button"] = ui.button(
            "✨ Generate Guide",
            on_click=generate_guide_action
        ).classes("w-full bg-black text-white mt-4 p-3 rounded-lg")

        state["generate_button"].disable()


    # wire up name input changes to update button state
    first_widget = state["widgets"].get("first_name")
    last_widget = state["widgets"].get("last_name")
    if first_widget:
        # NiceGUI input change hook; this uses event name used previously
        first_widget.on("update:modelValue", lambda _: update_generate_button_state())
    if last_widget:
        last_widget.on("update:modelValue", lambda _: update_generate_button_state())

    # expose some state for debugging if needed
    state["muscle_card_container"] = muscle_card_container
    state["stretch_card_container"] = stretch_card_container

    return state

@ui.page("/result")
def result_page():
    show_message("✅ Success", f"Guide saved as {filename}")
    ui.label(result_data["title"]).classes("text-xl font-bold")
    ui.markdown(result_data["content"]).classes("prose max-w-none")
    ui.button("⬅ Back", on_click=lambda: ui.navigate.to("/"))

def show_message(title: str, message: str):
    with ui.dialog() as dialog, ui.card():
        ui.label(title).classes("text-lg font-bold")
        ui.label(message).classes("mt-2")
        ui.button("OK", on_click=dialog.close).classes("mt-4")
    dialog.open()


# ===== Run App =====


def main():
    cfg = json.loads(CONFIG_JSON)
    build_ui_from_config(cfg)

    # ✅ Detect Jupyter/IPython
    if "ipykernel" in sys.modules:
        # Running inside Jupyter, avoid asyncio.run()
        ui.run(host="0.0.0.0", port=8081, native=False, reload=False, loop="asyncio")
    else:
        ui.run(host="0.0.0.0", port=8081, native=False, reload=False)

if __name__ in {"__main__", "__mp_main__"}:
    main()
