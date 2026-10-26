# frozen_string_literal: true
#
# Build-time wiki generator.
#
# Reads the plain-markdown folders below (which keep NO Jekyll front matter, so they
# stay portable / viewable on GitHub) and emits fully-styled al-folio pages at build
# time. Nothing in the source folders is modified, so the markdown can be edited freely
# and these pages regenerate on every build.
#
#   ai_research_wiki/**/*.md -> /wiki/<path>/      (README.md -> /wiki/)
#   deeplearning/**/*.md  -> /deeplearning/<path>/ (README.md -> /deeplearning/)
#
# Each page: layout `page`, title from the first "# H1", right-side TOC sidebar
# (bootstrap-toc), internal [..](*.md) links rewritten to permalinks, and math/code
# rendered by WikiRender (see wiki_render.rb, unit-tested separately).
require_relative 'wiki_render'
require 'pathname'

module WikiSite
  # source dir (relative to site.source) => { url prefix, display title for the index page }
  SOURCES = {
    'ai_research_wiki' => { 'prefix' => 'wiki', 'title' => 'AI Researcher Intern Wiki' },
    'deeplearning'     => { 'prefix' => 'deeplearning', 'title' => 'DL Fundamentals Wiki' }
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
      SOURCES.each do |dir, cfg|
        prefix     = cfg['prefix']
        wiki_title = cfg['title']
        root = site.in_source_dir(dir)
        next unless File.directory?(root)

        md_files = Dir.glob(File.join(root, '**', '*.md')).sort
        next if md_files.empty?

        # Wiki source folders are excluded from Jekyll's normal file copying.
        # Publish their images explicitly, preserving the source directory paths.
        Dir.glob(File.join(root, '**', '*')).each do |path|
          next unless File.file?(path) && path.match?(/\.(png|jpe?g|gif|svg|webp|avif)\z/i)

          relative = Pathname.new(path).relative_path_from(Pathname.new(site.source)).to_s
          site.static_files << Jekyll::StaticFile.new(site, site.source, File.dirname(relative), File.basename(relative))
        end

        md_files.each do |path|
          base = File.basename(path)
          relative = Pathname.new(path).relative_path_from(Pathname.new(root)).to_s
          stem = relative.sub(/\.md\z/i, '')
          index = base.casecmp('README.md').zero?
          slug = index ? File.dirname(relative) : stem
          out_dir = slug == '.' ? prefix : "#{prefix}/#{slug}"
          permalink = "/#{out_dir}/"

          link_map = build_link_map(root, prefix, File.dirname(path))
          h1, html = WikiRender.render(File.read(path), link_map)
          # index page uses the configured wiki name; chapters use their own H1
          title = index ? wiki_title : (h1.nil? || h1.empty? ? stem : h1)
          site.pages << WikiPage.new(site, out_dir, title, html, permalink)
        end
      end
    end

    private

    # Paths relative to the current page -> permalinks or published asset URLs.
    #   *.md          -> /<prefix>/<slug>/  (README -> /<prefix>/)
    #   images        -> /<dir>/<path>      (published explicitly above)
    def build_link_map(root, prefix, page_dir)
      map = {}
      dirname = File.basename(root)
      Dir.glob(File.join(root, '**', '*')).each do |path|
        next if File.directory?(path)
        name = File.basename(path)
        relative = Pathname.new(path).relative_path_from(Pathname.new(root)).to_s
        target = Pathname.new(path).relative_path_from(Pathname.new(page_dir)).to_s
        map[target] = if name =~ /\.md\z/i
                      slug = name.casecmp('README.md').zero? ? File.dirname(relative) : relative.sub(/\.md\z/i, '')
                      slug == '.' ? "/#{prefix}/" : "/#{prefix}/#{slug}/"
                    else
                      "/#{dirname}/#{relative}"
                    end
      end
      map
    end
  end
end
