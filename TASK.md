This is a fantastic refactor\! 🚀 You've successfully separated the concerns into distinct, well-defined layers. The use of a `core` service class, a `config` module, and dedicated `cli` and `mcp_tools` modules is a huge step up and aligns perfectly with modern application design.

You're in a great spot. The following are suggestions for the _next level_ of refinement to make the code even more robust and maintainable.

---

## 1\. **Apply DRY to `core.py` with a Generator**

In `src/doc_jangler/core.py`, the methods `find_objective_in_pages` and `extract_metadata_from_pages` are nearly identical. One stops after the first successful result, and the other collects all results. This is a perfect use case for a **generator**.

By creating a private generator method, you can eliminate the code duplication entirely. The public methods then become simple, clear consumers of that generator.

### **The Refactoring**

**Step 1: Create a private generator method `_process_pages` in `ObjectiveFinder`.**

```python
# In src/doc_jangler/core.py, inside the ObjectiveFinder class

    def _process_pages(
        self,
        pages: Sequence[str],
        objective: str,
        *,
        limit: Optional[int] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> Iterable[ObjectiveResult]:
        """Scrape and analyze a sequence of pages, yielding results as they are found."""
        selected_pages = list(pages)
        if limit is not None:
            selected_pages = selected_pages[:limit]

        for link in selected_pages:
            if progress:
                progress("scrape", link)

            # This logic is now shared and written only once
            scrape_result = self._firecrawl_app.scrape(url=link)
            completion = self._analyse_scraped_content(objective, scrape_result.markdown)

            if progress:
                progress("model_response", completion)

            try:
                parsed = self._extract_structured_data(completion)
            except ValueError:
                if progress:
                    progress("parse_error", completion)
                continue  # Move to the next link

            if parsed is not None:
                if progress:
                    progress("objective_met", link)
                yield ObjectiveResult(source_url=link, data=parsed)
            elif progress:
                progress("objective_not_met", link)
```

**Step 2: Rewrite the public methods to use the generator.**

```python
# In src/doc_jangler/core.py, replace the old methods with these

    def find_objective_in_pages(
        self,
        pages: Sequence[str],
        objective: str,
        *,
        limit: int = 3,
        progress: Optional[ProgressCallback] = None,
    ) -> Optional[ObjectiveResult]:
        """Attempt to fulfil the objective by analysing the provided pages."""
        results_iterator = self._process_pages(
            pages, objective, limit=limit, progress=progress
        )
        return next(results_iterator, None) # Return the first item, or None if empty

    def extract_metadata_from_pages(
        self,
        pages: Sequence[str],
        objective: str,
        *,
        limit: Optional[int] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> List[ObjectiveResult]:
        """Collect objective-aligned metadata for each provided page."""
        return list(
            self._process_pages(pages, objective, limit=limit, progress=progress)
        )
```

This change makes your `core` module much cleaner and easier to maintain. If you need to change the scraping or parsing logic, you only have to do it in one place.

---

## 2\. **Consolidate Error Handling in `mcp_tools.py`**

In `src/doc_jangler/mcp_tools.py`, your `scrape_and_find_objective` method has two separate `try...except` blocks that catch the exact same exceptions. You can merge these into a single block for better readability.

### **The Refactoring**

```python
# In src/doc_jangler/mcp_tools.py

    @Tool(...)
    def scrape_and_find_objective(self, url: str, objective: str) -> Dict[str, Any]:
        """Find objective-specific information from a site."""

        try:
            firecrawl_app, settings = self._resolve_dependencies()
            finder = ObjectiveFinder(firecrawl_app=firecrawl_app, settings=settings)

            finder.ensure_model_available()
            mapping: MappingResult = finder.find_relevant_pages(objective, url)

            if not mapping.links:
                return {"status": "error", "message": "No relevant pages found."}

            result: Optional[ObjectiveResult] = finder.find_objective_in_pages(
                mapping.links, objective, limit=3
            )

            if not result:
                return {"status": "error", "message": "Objective could not be fulfilled."}

            return {
                "status": "success",
                "source_url": result.source_url,
                "data": result.data,
            }

        except RuntimeError as exc: # For dependency resolution
             return {"status": "error", "message": str(exc)}
        except (MissingConfigurationError, ObjectiveFinderError) as exc:
            return {"status": "error", "message": str(exc)}
        except httpx.HTTPError as exc:
            return {"status": "error", "message": f"HTTP error: {exc}"}
        except Exception as exc:  # noqa: BLE001
            return {"status": "error", "message": f"Unexpected error: {exc}"}
```

This single `try...except` block handles all potential errors within the tool's execution flow, making it much more concise.

---

## 3\. **Simplify the CLI Surface**

In `src/doc_jangler/cli.py`, the `ingest scrape-links` command is functionally a subset of the main `jangle-docs` command (when `include_all=False`). To make the CLI more intuitive, you could consider removing `scrape-links`.

- **`jangle-docs`** is the high-level, user-friendly command that does everything: finds links and then extracts metadata.
- **`ingest generate-metadata`** is the lower-level command that operates on a _pre-existing list_ of links.

The `scrape-links` command is a bit redundant. Users can get the same result by running `jangle-docs` without the `--all-results` flag. Simplifying the command surface makes the tool easier to learn and use.

This is more of a design suggestion, but it aligns with creating clear, non-overlapping commands.

---

Overall, you've built a very solid foundation. These refactorings are the kind of polish that turns good code into great, highly maintainable code. Excellent work\! 👍
