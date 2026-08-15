# MiniMax H3 T2VA prompt writing

When a task requires a MiniMax H3 text-to-video-with-audio (T2VA) prompt, write the final prompt in English and return only these three fields in this exact order:

```text
integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ...
```

Follow these rules:

- Build a complete audiovisual timeline directly from the user's text. Do not add image-reference alignment instructions.
- Start with `[Shot 1]` and do not timestamp the first shot. For later cuts, use sequential shot numbers and strictly increasing `HH:MM:SS.mmm` timestamps within the requested duration.
- Describe composition, subject identity and appearance, environment, observable action, camera movement, and synchronized diegetic sound. Keep identity, clothing, objects, and spatial relationships consistent.
- Express camera movement naturally using motion type and, when useful, amplitude and speed.
- Put spoken content in `<d>[Language] exact dialogue</d>` and keep a speaker ID such as `(S1)` stable across shots.
- Write `overall_soundscape` as one concise paragraph covering ambience, physical sounds, and non-verbal human sounds. Do not repeat dialogue or music there.
- Write `non_diegetic_music` using instrumentation, tempo, rhythm, and dynamics. Use `N/A` when no background score is wanted.
- Match all timing and action density to the requested video duration. Do not output analysis, a preface, Markdown fences, or extra fields.

T2VA example:

```text
integrated_multimodal_description: [Shot 1] 2D anime sports-film style, a low medium-wide shot frames a young adult sprinter in navy track shorts and a white athletic top crouched in starting blocks on a sunlit red track. At the starter crack she drives forward with a strong arm swing and rising stride cadence as the camera tracks right with small amplitude at fast speed, keeping her profile sharp while lane markings blur behind her. Her ponytail and clothing move consistently with the acceleration, with no text or logos.
overall_soundscape: The starting mechanism snaps sharply, followed by rapid shoe strikes on rubber, controlled breathing, light fabric movement, and wind passing the runner.
non_diegetic_music: Fast percussion and bright strings maintain a steady pulse, rising briefly in volume as the runner reaches full acceleration.
```
