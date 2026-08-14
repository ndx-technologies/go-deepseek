package godeepseek

import "strconv"

type Error struct {
	HTTPStatusCode int
	Body           string
}

func (s *Error) Error() string { return strconv.Itoa(s.HTTPStatusCode) + " : " + s.Body }
