# Makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# this is to to autmagic gh-pages
GH_PAGES_SOURCES = docs qcodes CONTRIBUTING.rst LICENSE

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean genapi htmlfast

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf _auto
	rm -rf api/generated
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# generate api docs for instruments automatically
genapi:
	sphinx-apidoc  -o  _auto  -d 10 ../qcodes \
		../qcodes/instrument_drivers/american_magnetics\* \
		../qcodes/instrument_drivers/agilent \
		../qcodes/instrument_drivers/AimTTi \
		../qcodes/instrument_drivers/AlazarTech \
		../qcodes/instrument_drivers/basel \
		../qcodes/instrument_drivers/HP \
		../qcodes/instrument_drivers/ithaco \
		../qcodes/instrument_drivers/Keithley \
		../qcodes/instrument_drivers/keysight \
		../qcodes/instrument_drivers/Lakeshore \
		../qcodes/instrument_drivers/QDev/* \
		../qcodes/instrument_drivers/QuantumDesign/* \
		../qcodes/instrument_drivers/rigol/* \
		../qcodes/instrument_drivers/rohde_schwarz/* \
		../qcodes/instrument_drivers/stahl/*\
		../qcodes/instrument_drivers/stanford_research/* \
		../qcodes/instrument_drivers/signal_hound/* \
		../qcodes/instrument_drivers/tektronix/* \
		../qcodes/instrument_drivers/weinschel/* \
		../qcodes/instrument_drivers/yokogawa
	mkdir -p api/generated/
	cp _auto/qcodes.instrument_drivers.* api/generated/

# faster build by skipping execution of all notebooks
htmlfast: genapi
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -D nbsphinx_execute=never

# this is only left for backwards compatibility.
.PHONY: html-api
html-api: html

# this is only left for backwards compatibility.
.PHONY: htmlapi
htmlapi: html

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile genapi
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
