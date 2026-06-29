# frozen_string_literal: true
#
# Build-time wiki generator.
#
# Reads the plain-markdown folders below (which keep NO Jekyll front matter, so they
# stay portable / viewable on GitHub) and emits fully-styled al-folio pages at build
# time. Nothing in the source folders is modified, so the markdown can be edited freely
# and these pages regenerate on every build.
#
#   ai_research_wiki/*.md  ->  /wiki/<slug>/        (README.md -> /wiki/)
#   deeplearning/*.md      ->  /deeplearning/<slug>/ (README.md -> /deeplearning/)
#
# Each page: layout `page`, title from the first "# H1", right-side TOC sidebar
# (bootstrap-toc), internal [..](*.md) links rewritten to permalinks, and math/code
# rendered by WikiRender (see wiki_render.rb, unit-tested separately).
require_relative 'wiki_render'

module WikiSite
  # source dir (relative to site.source) => URL prefix
  SOURCES = {
    'ai_research_wiki' => 'wiki',
    'deeplearning'     => 'deeplearning'
  }.freeze

  class WikiPage < Jekyll::PageWithoutAFile
    def initialize(site, dir, title, html, permalink)
      super(site, site.source, dir, 'index.html')
      # {% raw %} guard: the pre-rendered HTML is final; never let Liquid touch it
      # (protects any stray {{ }} / {% %} that could appear inside code samples).
      self.content = "{% raw %}\n#{html}\n{% endraw %}"
      self.data = {
        'layout'         => 'page',
        'title'          => title,
        'permalink'      => permalink,
        'toc'            => { 'sidebar' => 'right' },
        'wiki_generated' => true
      }
    end
  end

  class Generator < Jekyll::Generator
    safe true
    priority :low

    def generate(site)
      SOURCES.each do |dir, prefix|
        root = site.in_source_dir(dir)
        next unless File.directory?(root)

        md_files = Dir.glob(File.join(root, '*.md')).sort
        next if md_files.empty?

        link_map = build_link_map(root, prefix)

        md_files.each do |path|
          base = File.basename(path)
          stem = base.sub(/\.md\z/i, '')
          index = base.casecmp('README.md').zero?
          out_dir   = index ? prefix : "#{prefix}/#{stem}"
          permalink = index ? "/#{prefix}/" : "/#{prefix}/#{stem}/"

          title, html = WikiRender.render(File.read(path), link_map)
          title = stem if title.nil? || title.empty?
          site.pages << WikiPage.new(site, out_dir, title, html, permalink)
        end
      end
    end

    private

    # filename -> permalink, for rewriting internal links inside the markdown.
    #   *.md          -> /<prefix>/<slug>/  (README -> /<prefix>/)
    #   other assets  -> /<dir>/<name>      (Jekyll copies them verbatim from the folder)
    def build_link_map(root, prefix)
      map = {}
      dirname = File.basename(root)
      Dir.glob(File.join(root, '*')).each do |path|
        next if File.directory?(path)
        name = File.basename(path)
        map[name] = if name =~ /\.md\z/i
                      name.casecmp('README.md').zero? ? "/#{prefix}/" : "/#{prefix}/#{name.sub(/\.md\z/i, '')}/"
                    else
                      "/#{dirname}/#{name}"
                    end
      end
      map
    end
  end
end
