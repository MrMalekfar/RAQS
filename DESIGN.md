# RAQS desktop design system

## Purpose and composition

Mode: Operate. The workbench supports a sustained scientific task, not a promotional dashboard. The hierarchy is experiment configuration, physical inspection, execution, then measurement review.

The compact command bar contains Run Diagnostics, Start Calculation and Stop Calculation. There is no in-content application title. Eligibility and notifications live in the bottom status area. The visible eligibility line separates an editable draft, a parsed profile, an executable request and a verified result. A four-item workspace navigation selects Configure, Inspect physics, Run monitor and Results; it does not nest page tabs inside page tabs.

Configure presents one scientific section at a time. Probe laser, coupling laser, RF oscillator, RF signal, prepared states, array/acquisition and thermal/relaxation settings are named by research task. Live JSON remains alongside the form in a resizable splitter. Add/replace and Remove effect are normal configuration actions; Advanced structure is the fallback for arbitrary nested data.

## Visual tokens and behavior

- Canvas: #f3f5f8; primary text: #17293d; secondary text: #4e6074.
- Primary action and selected navigation: #235f91 with white text. Hover: #194c78.
- White input/result surfaces; #bdc9d5 borders; #e6ecf3 navigation surface.
- Error surface: #fae8e5 with #902a23 text. Invalid fields use a #ad3434 border and an adjacent textual correction. Color is never the only status indication.
- Typography: DejaVu Sans, 10 pt inherited body; no oversized application heading and 12 pt page titles. DejaVu Sans Mono is reserved for source JSON and logs. Fonts are loaded from the already-installed Matplotlib distribution.
- Spacing: 10 px horizontal outer margin, 8 px top margin, 6 px between logical groups. Controls have 7 px vertical padding, 4 px corner radii and visible keyboard focus.
- Standard Qt controls retain selection, keyboard traversal and accessibility roles. No decorative animation, gradients, invented metric cards or extra icon package is used.
- Initial window: 1380×880 logical pixels. Minimum: 1000×680. Scientific forms, reports, tables and figures scroll; splitters let users trade form/preview space.

## Scientific input rules

Numbers remain editable text so small values and full decimal/scientific notation are not rounded by a spinbox. Unit labels are presentation metadata; scientific constraints remain in the canonical parser. F/mF and j/mJ retain their notation. Import rebuilds controls and preserves unhandled sections. Invalid drafts disable execution of the previous valid input and invalidate displayed diagnostics/results.

## Results and execution

Results lead with the scientific figure. Measurement-specific Summary and Technical report remain adjacent tabs. IF-only never presents an estimated angle. Figure selection, Fit Page (default), Fit Width, 100/150/200% image zoom and safe PNG export use saved scientific figures; this is image navigation, not recomputation or interactive data analysis. The original report remains available in Technical report.

The monitor shows real worker stages, measured elapsed time and the tail of a preserved log. Saved-result verification runs in a background task; unverified data are not displayed as a completed result. No predicted completion percentage is offered. Windows Stop uses a gated, owned process group; interrupted data are not accepted as completed output.

## Verification boundaries

Filled configuration, invalid-draft, diagnostics, measurement-summary and figure views are inspected in a bounded render review at normal and compact sizes. Qt event-driven tests exercise controls and actual ARC/spawn integration. Native screenshot capture and pointer interaction, multiple Windows DPI settings and assistive technology coverage remain separate checks; offscreen success is not proof of those capabilities.

Focus Figure reclaims navigation/metadata/status space while keeping the figure controls and global calculation commands reachable; Escape restores the layout. Live logs follow the newest line unless the user disables following. Advanced Structure editors use explicit geometry, readable padding, 30px-class rows and a blue border so selected text is not clipped by normal input padding.
