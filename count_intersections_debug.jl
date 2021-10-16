function count_intersections(marbles, lineslope=nothing, line=[])
    @info "="^85
    @info "Marbles (before)", map(Tuple, marbles)
    if isempty(marbles)
        @info "Terminate."
        return line
    end

    if isempty(line)
        # pick anchor from list of all marbles
        anchor_marbles = marbles
    else
        # if `line` is provided, then anchor at the 2nd added (i.e. the "candidate")
        anchor_marbles = [line[end]]
        marbles = vcat(line[end], marbles)
    end
    @info "Anchor marbles", map(Tuple, anchor_marbles)
    @info "Marbles (after)", map(Tuple, marbles)

    # for every "anchor" marble
    for (i, anchor) in enumerate(anchor_marbles)
        for (j, other) in enumerate(marbles)
            if i ≠ j
                # pick another marble, draw a line to it (i.e., get slope `m`)
                @info Tuple(anchor), Tuple(other)
                slope_a2o = slope(anchor, other)
                @info slope_a2o
                @assert anchor != other
                anchor2other = [anchor, other]
                @info "anchor2other", Tuple.(anchor2other)
                if isnothing(lineslope) || lineslope == slope_a2o
                    if length(line) ≤ 2
                        @info "line = anchor2other"
                        line = anchor2other
                    else
                        @info "line = vcat(line, other)"
                        line = vcat(line, other)
                    end
                    # for every candidate (non-same and not currently added)
                    # draw a line from anchor-to-candidate
                    # (bypassing the "other" for now)
                    for candidate in marbles
                        if candidate ∉ [anchor, other]
                            anchor2cand = [anchor, candidate]
                            # if the two lines are parallel (given same anchor),
                            # —> add the 2nd marble to the current longest line
                            slope_a2c = slope(anchor, candidate)
                            if slope_a2o == slope_a2c
                                @info "FOUND ONE!"
                                @info Tuple(anchor), Tuple(other), Tuple(candidate)
                                if length(line) ≤ 2
                                    line = [anchor, other, candidate]
                                end
                                marbs = setdiff(marbles, line)
                                # recurse, reducing the size of marbles::Vector and 
                                # anchoring at "other", keeping
                                # —> the [other, candidate] as the "running line"
                                return count_intersections(marbs, slope_a2c, line) 
                            end
                        end
                    end
                end
            end
        end
    end
    return line
end

test_marbles = [State(1,1), State(1,2), State(1,3), State(1,4)]
count_intersections(test_marbles)
