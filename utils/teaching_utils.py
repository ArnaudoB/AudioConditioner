TEACHER_PROMPT = """You are a JSON generator. Return ONLY valid JSON (no markdown, no comments, no extra text).

Task: Convert the given piece of story into a JSON that will be used to prompt a text-to-music model for an instrumental clip that will serve as the soundtrack for the scene described in the story.

Rules:
- Output must conform exactly to this schema (all keys required):
{{
  "mood": [<string>],
  "energy": <float 0..1>,
  "valence": <float 0..1>,
  "tempo_bpm": <int 50..180>,
  "key_mode": <"major"|"minor"|"ambiguous">,
  "harmonic_tension": <float 0..1>,
  "texture_density": <float 0..1>,
  "instrumentation": [<string>],
  "rhythm_style": <string>,
  "structure": <string>,
  "production_style": [<string>],
  "dynamics_profile": <string>
}}

- Choose moods ONLY from (you can pick several moods): ["dark","epic","ominous","tragic","tense","mysterious","melancholic","nostalgic","peaceful","warm","gentle","uplifting","triumphant","joyful","hopeful","anxious","aggressive","dreamy","majestic","suspenseful"].
- Choose instrumentation ONLY from: ["low strings","strings","brass","woodwinds","war drums","taiko","timpani","cinematic percussion","piano","acoustic guitar","electric guitar","bass","synth bass","pads","ambient textures","lofi drums","trap drums","shakers","bells","marimba","harp","flute","organ"].
- Choose rhythm_style ONLY from: ["sparse","pulsing","steady","syncopated","driving","minimal","floating"].
- Choose structure ONLY from: ["loopable","slow build then climax","constant evolving texture","intro then drop then resolve","rise and fall"].
- Choose production_style ONLY from (you can pick several styles) : ["cinematic","orchestral","ambient","lofi","electronic","modern","vintage","wide stereo","reverb heavy","clean studio"].
- Choose dynamics_profile ONLY from: ["gradual crescendo","constant intensity","sudden impact near end","soft throughout","build and release"].
- Must be instrumental (do not mention vocals).
- Pick values that best match the SCENE.

SCENE: {scene_text}
"""

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

MOOD_LIST = ["dark","epic","ominous","tragic","tense","mysterious","melancholic","nostalgic","peaceful","warm","gentle","uplifting","triumphant","joyful","hopeful","anxious","aggressive","dreamy","majestic","suspenseful"]
INSTRUMENTATION_LIST = ["low strings","strings","brass","woodwinds","war drums","taiko","timpani","cinematic percussion","piano","acoustic guitar","electric guitar","bass","synth bass","pads","ambient textures","lofi drums","trap drums","shakers","bells","marimba","harp","flute","organ"]
RHYTHM_STYLE_LIST = ["sparse","pulsing","steady","syncopated","driving","minimal","floating"]
STRUCTURE_LIST = ["loopable","slow build then climax","constant evolving texture","intro then drop then resolve","rise and fall"]
PRODUCTION_STYLE_LIST = ["cinematic","orchestral","ambient","lofi","electronic","modern","vintage","wide stereo","reverb heavy","clean studio"]
DYNAMICS_PROFILE_LIST = ["gradual crescendo","constant intensity","sudden impact near end","soft throughout","build and release"]
KEY_MODE_LIST = ["major", "minor", "ambiguous"]
TEMPO_RANGE = (50, 180)