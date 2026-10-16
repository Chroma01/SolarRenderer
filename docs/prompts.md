# Prompts and first frames


SolarRenderer was trained on one six-part prompt form, and a prompt in that form is the single biggest lever on
quality; a short free-form sentence works much worse. [`skills/solarrenderer-caption/`](../skills/solarrenderer-caption)
is a standard agent skill (`SKILL.md` and three helpers) that teaches a coding agent such as Codex or Claude Code to
write it:

```
subject_definitions:   what each input is            (fixed sentences, from conditions.py)
summary:               task type + one paragraph     (who does what, how the camera moves)
retention_analysis:    what is taken from each input (fixed sentences)
detailed_description:  350-500 words                 (materials, light, weather, characters, events, in order)
overall_soundscape:    ambience
non_diegetic_music:    N/A
```

Give the skill folder to your agent together with the structure video (and the first frame, if you have one), and
ask it to write `prompt.txt` in the look you want. It will:

1. watch the structure video: the number of characters, their facing, the layout and the camera path;
2. pick the structure sentence: one of the six trained types in `conditions.DEFINE` word for word, or, for any
   other structure, `custom_condition(looks_like, not_shown)`, a plain description of what the clip looks like and
   what of its look must not appear;
3. write the free parts, spending the words on what the structure leaves open;
4. assemble the prompt with `conditions.assemble(free, inputs, condition)` and validate it with
   `python skills/solarrenderer-caption/check.py prompt.txt`.

**No first frame?** The skill also covers making one with an image model (Codex's image generation, or any model
that edits a reference image). Frame 0 of the structure video is the reference; the image prompt keeps its camera,
layout, characters, poses and facing, and replaces the grey surfaces with the look the prompt describes:

```bash
F=skills/solarrenderer-caption/first_frame.py
python $F frame0 structure.mp4 frame0.png      # the reference image
python $F prompt prompt.txt                     # the image prompt (--stylised for toon, pixel art, ...)
# generate a 16:9 image from frame0.png with that prompt, then:
python $F fit generated.png first.png           # crop and scale to 1344x768
```

To restyle a clip, keep the structure video, write the prompt for the new world, and make a first frame in that world.
