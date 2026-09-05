# NanoLocz v1.0 Release Checklist

## Pre-Release Quality Gates

### Code Quality
- [x] All tests passing (473 passed, 112 skipped for CuPy)
- [x] No critical or high-severity bugs open
- [x] Code follows project style guidelines
- [x] All functions have docstrings
- [x] Type hints present on public APIs

### Test Coverage
- [x] Unit tests: 473+ passing
- [x] Integration tests: CLI, Napari plugin
- [x] GPU tests skip gracefully when CuPy unavailable
- [x] Test fixtures included in repository

### Documentation
- [x] README.md complete with installation instructions
- [x] API documentation (docstrings)
- [x] CLI help text (`nanolocz --help`)
- [x] Example workflows in `examples/`
- [x] Benchmark documentation in `docs/benchmarks/`
- [x] Migration guide from MATLAB

### Security & Compliance
- [x] LICENSE file present (GPL-3.0)
- [x] NOTICE.md with attributions
- [x] SECURITY.md present
- [x] CODE_OF_CONDUCT.md present
- [x] All dependencies have compatible licenses

## Distribution Artifacts

### PyPI Package
- [ ] Version updated to 1.0.0 in pyproject.toml
- [ ] Source distribution builds: `python -m build`
- [ ] Wheel builds successfully
- [ ] TestPyPI upload successful
- [ ] Production PyPI upload: `nanolocz==1.0.0`
- [ ] Package verified after upload: `pip install nanolocz==1.0.0`

### Docker Images
- [ ] CPU image builds: `docker build -t nanolocz:cpu .`
- [ ] GPU image builds: `docker build -f Dockerfile.gpu -t nanolocz:gpu .`
- [ ] CPU image tested: runs CLI and Napari
- [ ] GPU image tested: CUDA available, CuPy works
- [ ] Pushed to Docker Hub: `stavros/nanolocz:v1.0.0`
- [ ] Tagged as latest: `stavros/nanolocz:latest`
- [ ] GPU tagged: `stavros/nanolocz:gpu-v1.0.0`

### Conda Packages
- [ ] Conda recipe created in `conda.recipe/`
- [ ] Local build tested: `conda-build conda.recipe/`
- [ ] conda-forge PR submitted
- [ ] Package available on conda-forge

### GitHub Release
- [ ] Git tag created: `git tag v1.0.0`
- [ ] Tag pushed: `git push origin v1.0.0`
- [ ] GitHub Release created with changelog
- [ ] Release notes include installation instructions
- [ ] Assets attached (optional: wheels, source tarball)

## Final Verification

### Clean Environment Tests
- [ ] Fresh virtualenv: `pip install nanolocz` works
- [ ] CLI smoke test: `nanolocz --help`
- [ ] Napari plugin loads: `napari -w nanolocz`
- [ ] Basic workflow completes end-to-end

### Hardware Configurations
- [ ] Tested on CPU-only system
- [ ] Tested on GPU system (if available)
- [ ] Benchmarks run successfully

### Documentation Final Check
- [x] STATUS.md updated to mark Phase 4 COMPLETE
- [x] SPEC/tasks.md shows all cards done
- [x] SESSIONS/ has handoff for NL-43
- [x] CHANGELOG.md or GitHub Releases has v1.0.0 entry

## Post-Release Tasks

### Announcement
- [ ] Email announcement to relevant mailing lists
- [ ] Social media posts (Twitter, LinkedIn)
- [ ] Update project website
- [ ] Notify collaborators and stakeholders

### Monitoring
- [ ] Watch issue tracker for bug reports
- [ ] Monitor PyPI download statistics
- [ ] Track Docker pull counts
- [ ] Gather user feedback

### Planning
- [ ] Collect feature requests for v1.1.0
- [ ] Plan NL-41b advanced features
- [ ] Schedule NL-54 BioAFMviewer validation

---

## Sign-off

**Release Manager:** ___________________  
**Date:** ___________________  
**Version:** 1.0.0  

**Quality Assurance:** ___________________  
**Date:** ___________________

---

*Complete this checklist before tagging v1.0.0. All items must be checked.*
